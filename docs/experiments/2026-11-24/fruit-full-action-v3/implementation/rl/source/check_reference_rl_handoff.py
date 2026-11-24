"""Audit the fruit pose checkpoint in the RLinf actor before policy updates."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from openpi.shared import normalize
import torch

from pi05_runtime import build_model, file_sha256, restore_trainable, trainable_snapshot
from ppo_pi05 import recompute
from train_reference_aligned_fruit import (AlignedPolicy, DIM, EXECUTE, FruitEpisode, HORIZON, PADDED_DIM,
                                           evaluate, paired_episodes)


class RLPolicy(AlignedPolicy):
    def __init__(self, model, stats, fork, **settings):
        super().__init__(model, stats, fork, **settings)
        model.setup_transforms([], [self.outputs])

    def infer(self, raw):
        with torch.inference_mode():
            obs, _, row = self.encode(raw)
            _, detail = self.model._predict_eval(obs, rng=self.generator, noise=self.reference_sample_noise())
        return self.outputs({'actions': detail['model_actions'][0].float().cpu().numpy(), 'state': row['state']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'run', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--episode', required=True, type=Path, action='append',
                        help='Repeat to check real observations from several tasks')
    parser.add_argument('--paired-episodes', type=Path, help='Check deterministic rollouts after loading')
    parser.add_argument('--eval-seed', action='append', type=int)
    args = parser.parse_args()
    if args.eval_seed and not args.paired_episodes:
        parser.error('--eval-seed requires --paired-episodes')
    checkpoint_path = args.run / 'sft.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    base_hash = file_sha256(args.base)
    if meta.get('base_sha256', base_hash) != base_hash:
        raise ValueError('The RL handoff received a different base checkpoint')
    stats_data = (args.run / 'norm_stats.json').read_bytes()
    if (hashlib.sha256(stats_data).hexdigest() != meta['norm_stats_sha256']
            or meta['horizon'] != HORIZON or meta['execute'] != EXECUTE):
        raise ValueError('Unexpected checkpoint interface or normalization')
    settings = dict(shortest_euler=meta.get('shortest_euler', False), discrete_state=meta.get('discrete_state', False),
                    reference_noise=bool(meta.get('reference_model_revision')))
    stats = normalize.deserialize_json(stats_data.decode())
    options = {'lora_rank': meta['actor_lora_rank'], 'vlm_lora_rank': meta['vlm_lora_rank'],
               'action_horizon': HORIZON, 'action_env_dim': DIM, 'discrete_state_input': settings['discrete_state'],
               'train_action_heads': meta.get('train_action_heads', True),
               'train_full_action_expert': meta.get('train_full_action_expert', False)}
    episodes = [FruitEpisode(path) for path in args.episode]
    if len({(episode.task, episode.seed) for episode in episodes}) != len(episodes):
        raise ValueError('Each task/seed may be checked only once')
    windows = [(episode, frame) for episode in episodes
               for frame in (0, len(episode.state) // 2, len(episode.state) - 1)]
    noise = torch.randn(1, HORIZON, PADDED_DIM, device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(2401))
    sft_model, _ = build_model(args.base, 'cuda', EXECUTE, **options)
    restore_trainable(sft_model, checkpoint['policy'])
    sft = AlignedPolicy(sft_model, stats, args.fork, **settings)
    reference = []
    with torch.inference_mode():
        for episode, frame in windows:
            raw = episode.row(frame)
            raw.pop('actions')
            obs, _, row = sft.encode(raw)
            sample = sft_model.sample_actions(obs, noise=noise)
            absolute = sft.outputs({'actions': sample[0].float().cpu().numpy(), 'state': row['state']})['actions']
            reference.append((sample.cpu(), absolute[:EXECUTE]))
    del sft_model, sft, obs, sample
    gc.collect()
    torch.cuda.empty_cache()
    rl_model, _ = build_model(args.base, 'cuda', EXECUTE, rl=True, **options)
    restore_trainable(rl_model, checkpoint['policy'])
    unchanged = trainable_snapshot(rl_model, actor_only=True)
    if any(not torch.equal(unchanged[key], value) for key, value in checkpoint['policy'].items()):
        raise ValueError('Loading the RL actor changed policy parameters')
    rl = RLPolicy(rl_model, stats, args.fork, **settings)
    metrics = []
    with torch.inference_mode():
        for (episode, frame), (normal, absolute) in zip(windows, reference, strict=True):
            raw = episode.row(frame)
            raw.pop('actions')
            inputs, _, encoded = rl.encode(raw)
            actions, detail = rl_model._predict_eval(inputs, noise=noise)
            model_actions = detail['model_actions'].cpu()
            torch.testing.assert_close(model_actions, normal, atol=0, rtol=0)
            # RLinf returns float32 actions. The rollout adapter keeps the shared NumPy decode.
            native_actions = actions[0].cpu().numpy()
            shared_actions = rl.outputs({'actions': model_actions[0].float().numpy(),
                                         'state': encoded['state']})['actions'][:EXECUTE]
            np.testing.assert_array_equal(shared_actions, absolute)
            _, rollout = rl_model._predict_train(inputs, noise=noise, compute_values=True,
                                                 rng=torch.Generator(device='cuda').manual_seed(2402))
            forward = rollout['forward_inputs']
            transition = {'chain': forward['chains'].cpu(), 'denoise_inds': forward['denoise_inds'].cpu(),
                          'executed': EXECUTE, 'row': raw}
            # The PPO recomputation helper uses the original two-item interface.
            class PPOEncoding:
                def encode(self, observation):
                    obs, target, _ = rl.encode(observation)
                    return obs, target
            logprob, _ = recompute(rl_model, PPOEncoding(), transition)
            old = rollout['prev_logprobs'][0, :EXECUTE].sum()
            error = float(abs(logprob - old))
            if error > 0.05:
                raise ValueError(f'Actor logprob diverges by {error:g}')
            metrics.append(dict(task=episode.task, seed=episode.seed, frame=frame, normalized_prediction_max_error=0.,
                                absolute_prediction_max_error=0.,
                                native_float32_decode_max_error=float(np.max(np.abs(native_actions - absolute))),
                                replay_logprob_abs_error=error))
    result = dict(checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  cases=[dict(task=episode.task, seed=episode.seed) for episode in episodes],
                  base_sha256=base_hash, norm_stats_sha256=meta['norm_stats_sha256'],
                  render_samples=meta.get('render_samples'),
                  executed=EXECUTE, horizon=HORIZON, settings=settings, measurements=metrics, optimizer_updates=0)
    result.update(readiness_diagnostic_only=True, training_scope=meta.get('training_scope'),
                  eligible_as_final_model=False, train_full_action_expert=meta.get('train_full_action_expert', False))
    rl.provenance = dict(sft_sha256=result['checkpoint_sha256'], base_sha256=base_hash,
                         norm_stats_sha256=meta['norm_stats_sha256'],
                         reference_model_revision=meta.get('reference_model_revision'))
    if args.paired_episodes:
        expected = paired_episodes(args.paired_episodes)
        rollouts = args.output.with_suffix('')
        seeds = args.eval_seed or sorted(expected)
        if len(set(seeds)) != len(seeds) or not set(seeds) <= expected.keys():
            raise ValueError('Requested seeds must be unique and present in the SFT rollouts')
        evaluate(rl, rollouts, expected, 600, False, 'rl-before-update', seeds=seeds,
                 render_samples=meta.get('render_samples'))
        measured = paired_episodes(rollouts / 'episodes.jsonl')
        for seed, actual in measured.items():
            for key in ('success', 'steps', 'action_sha256', 'ik_nonconverged_steps'):
                if actual[key] != expected[seed][key]:
                    raise RuntimeError(f'RL handoff changed the {key} for seed {seed}')
        result['matched_rollouts'] = len(measured)
        result['matched_successes'] = sum(r['success'] for r in measured.values())
        result['matched_seeds'] = list(measured)
        print(f'All {len(measured)} closed-loop trajectories matched the SFT policy exactly.', flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
