"""Measure sampler sensitivity on stored expert observations, without policy updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_pose_imitation import action_metrics
from pi05_runtime import file_sha256, restore_trainable
from reference_pose_lerobot import load_pose_split
from train_reference_aligned_fruit import EXECUTE, HORIZON, PADDED_DIM, load_aligned_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'run', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', default='select_fruit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    episodes = [ep for ep in load_pose_split(args.dataset, 'train')
                if (ep.task, ep.seed) == (args.task, args.seed)]
    if len(episodes) != 1:
        parser.error('Select exactly one existing training demonstration')
    if args.output.exists():
        parser.error('Use a fresh output file')
    episode, = episodes
    closed = np.flatnonzero(episode.action[:, 6] < 0.1)
    frames = sorted({0, *[int(closed[i]) for i in (0, len(closed) // 2, -1)]}) if len(closed) else [0]
    policy = load_aligned_policy(args.run, args.base, args.fork, stage='base')
    checkpoint = torch.load(args.run / 'sft.pt', map_location='cpu', weights_only=True)
    report = dict(kind='stored_training_observation_sampler_audit_not_closed_loop',
                  task=episode.task, seed=episode.seed, frames=frames,
                  source_sha256=file_sha256(__file__),
                  checkpoint_sha256=file_sha256(args.run / 'sft.pt'),
                  dataset_verification_sha256=file_sha256(args.dataset / 'verification.json'),
                  policy_updates=0, policy_rollouts=0,
                  noise_note='JAX noise is reset to the case seed at each stored observation; '
                             'Torch noise uses the sparse-validation seed rule. '
                             'Neither sequence represents a full rollout.',
                  records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for stage in ('base', 'sft'):
        if stage == 'sft':
            restore_trainable(policy.model, checkpoint['policy'])
        for frame in frames:
            raw = episode.row(frame)
            target = raw.pop('actions').copy()[:min(EXECUTE, len(episode.state) - frame)]
            observation, labels, encoded = policy.encode(raw)
            if labels is not None:
                raise RuntimeError('Action labels must not enter inference')
            policy.reset_rng(episode.seed)
            noises = {'jax_case_seed': policy.reference_sample_noise(),
                      'torch_validation_seed': torch.randn(
                          (1, HORIZON, PADDED_DIM), device='cuda',
                          generator=torch.Generator(device='cuda').manual_seed(500 + frame)),
                      'zero': torch.zeros((1, HORIZON, PADDED_DIM), device='cuda')}
            if noises['jax_case_seed'] is None:
                raise RuntimeError('Expected the published-reference noise interface')
            for sampler, noise in noises.items():
                for denoise_steps in (10, 30):
                    with torch.inference_mode():
                        result = policy.model.sample_actions(observation, num_steps=denoise_steps,
                                                             noise=noise.clone())
                    prediction = policy.outputs({'actions': result[0].float().cpu().numpy(),
                                                 'state': encoded['state']})['actions'][:len(target)].copy()
                    if not np.isfinite(prediction).all():
                        raise RuntimeError('Non-finite policy action')
                    row = dict(stage=stage, frame=frame, sampler=sampler, denoise_steps=denoise_steps,
                               target_gripper=target[:, 6].tolist(),
                               predicted_gripper=prediction[:, 6].tolist(),
                               **action_metrics(prediction, target))
                    report['records'].append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='complete', records=len(report['records']))), flush=True)


if __name__ == '__main__':
    main()
