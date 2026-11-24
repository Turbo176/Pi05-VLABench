"""Run PPO or a zero-update rollout audit with a pose-aligned π0.5 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from openpi.shared import normalize
import torch

from check_reference_rl_handoff import RLPolicy
from evaluate_vlabench_reference import TASK, build_episode, load_upstream_adapter
from pi05_runtime import actor_delta, build_model, file_sha256, restore_trainable, trainable_snapshot
from ppo_pi05 import GAMMA, GAE_LAMBDA, PPO_CLIP, fruit_potential, recompute, update
from select_fruit import open_episode
from train_reference_aligned_fruit import DIM, EXECUTE, HORIZON


class ActorEncoding:
    def __init__(self, policy):
        self.policy = policy

    def encode(self, row):
        observation, labels, _ = self.policy.encode(row)
        return observation, labels


def actor_change_evidence(current, original):
    """Measure actual VLA changes separately from the added value head."""
    groups = {}
    for name, before in original.items():
        delta = current[name].float() - before.float()
        group = ('action_heads' if not name.startswith('llm.') else
                 'vlm_lora' if '.0.q_lora_' in name else
                 'action_expert_lora' if '_lora_' in name else 'action_expert_full')
        row = groups.setdefault(group, dict(tensors=0, changed_tensors=0, changed_elements=0,
                                            max_abs_delta=0.0, squared_delta=0.0))
        changed = int(torch.count_nonzero(delta))
        row['tensors'] += 1
        row['changed_tensors'] += int(changed > 0)
        row['changed_elements'] += changed
        row['max_abs_delta'] = max(row['max_abs_delta'], float(delta.abs().max()))
        row['squared_delta'] += float(delta.square().sum())
    for row in groups.values():
        row['delta_l2'] = row.pop('squared_delta') ** 0.5
    return groups


class PPOClient:
    def __init__(self, policy, env, rng, shaping):
        self.policy, self.env, self.rng = policy, env, rng
        self.shaping = shaping
        self.potential = fruit_potential(env) if shaping else 0.0
        self.transitions = []
        self.pending = None

    def finish_chunk(self, executed, solved):
        if self.pending is None or not 1 <= executed <= EXECUTE:
            raise RuntimeError('Expected a sampled action chunk and its executed length')
        transition, logprobs = self.pending
        potential = fruit_potential(self.env) if self.shaping else 0.0
        reward = float(solved) + self.shaping * (potential - self.potential)
        transition.update(executed=executed, reward=reward, old_logp=float(logprobs[:executed].sum()))
        self.transitions.append(transition)
        self.potential = potential
        self.pending = None

    @torch.no_grad()
    def infer(self, row):
        if self.pending is not None:
            self.finish_chunk(EXECUTE, False)
        obs, _, encoded = self.policy.encode(row)
        _, sample = self.policy.model._predict_train(obs, noise=None, rng=self.rng, compute_values=True)
        forward = sample['forward_inputs']
        self.pending = (dict(row=row, chain=forward['chains'].cpu(), denoise_inds=forward['denoise_inds'].cpu(),
                             value=float(sample['prev_values'].item())), sample['prev_logprobs'][0].cpu())
        actions = sample['model_actions'][0].float().cpu().numpy()
        return self.policy.outputs({'actions': actions, 'state': encoded['state']})


def collect(policy, adapter, seed, rng, controls, shaping, *, render_samples=None):
    random_state = random.getstate()
    try:
        env = open_episode(seed, TASK, render_samples=render_samples)
    finally:
        random.setstate(random_state)
    with env:
        client = PPOClient(policy, env, rng, shaping)
        result, rollout = build_episode(env, adapter(client, replan_steps=EXECUTE), seed, controls, False)
        for _ in rollout:
            pass
        client.finish_chunk(result['steps'] - len(client.transitions) * EXECUTE, result['success'])
    # The evaluation budget defines a finite-horizon task. Its final chunk has no future value.
    next_value = advantage = 0.0
    for transition in reversed(client.transitions):
        delta = transition['reward'] + GAMMA * next_value - transition['value']
        advantage = delta + GAMMA * GAE_LAMBDA * advantage
        transition['advantage'] = advantage
        transition['return'] = advantage + transition['value']
        next_value = transition['value']
    return client.transitions, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'fork', 'sft-run', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--shaping', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=372)
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate of the trainable VLA actor parameters')
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    args = parser.parse_args()
    if (not 1 <= args.max_steps <= 600 or args.iterations < 0 or not 0 <= args.shaping <= 1
            or not 0 < args.lr <= 1 or not 0 < args.critic_lr <= 1
            or len(args.seeds) != len(set(args.seeds)) or any(not 1301 <= s <= 1399 for s in args.seeds)):
        parser.error('Use distinct PPO training starts in 1301–1399 and valid rollout budgets')
    if args.iterations and len(args.seeds) < 2:
        parser.error('PPO training requires at least two rollouts per iteration')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cuda_rng = torch.Generator(device='cuda').manual_seed(args.seed)
    checkpoint_path = args.sft_run / 'sft.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    base_hash = file_sha256(args.base)
    if metadata.get('base_sha256', base_hash) != base_hash:
        raise ValueError('PPO received a different base checkpoint from the SFT parent')
    stats_bytes = (args.sft_run / 'norm_stats.json').read_bytes()
    stats_hash = hashlib.sha256(stats_bytes).hexdigest()
    if (metadata['norm_stats_sha256'] != stats_hash or metadata['horizon'] != HORIZON
            or metadata['execute'] != EXECUTE):
        raise RuntimeError('The SFT checkpoint has a different policy interface')
    settings = dict(shortest_euler=metadata['shortest_euler'], discrete_state=metadata['discrete_state'])
    model, _ = build_model(args.base, 'cuda', EXECUTE, rl=True, action_env_dim=DIM, action_horizon=HORIZON,
                           lora_rank=metadata['actor_lora_rank'], vlm_lora_rank=metadata['vlm_lora_rank'],
                           discrete_state_input=settings['discrete_state'],
                           train_action_heads=metadata.get('train_action_heads', True))
    restore_trainable(model, checkpoint['policy'])
    policy = RLPolicy(model, normalize.deserialize_json(stats_bytes.decode()), args.fork, **settings)
    encoding = ActorEncoding(policy)
    adapter = load_upstream_adapter(args.fork)
    original = trainable_snapshot(model, actor_only=True)
    names = list(original)
    groups = [dict(params=[p for name, p in model.named_parameters()
                           if p.requires_grad and name in names], lr=args.lr),
              dict(params=[p for name, p in model.named_parameters()
                           if p.requires_grad and name.startswith('value_head.')], lr=args.critic_lr)]
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    total = 0
    total_optimizer_steps = 0
    with (args.output / 'rollouts.jsonl').open('x', encoding='utf-8') as stream:
        for iteration in range(max(args.iterations, 1)):
            start = time.monotonic()
            samples, successes = [], 0
            iteration_controls = 0
            for seed in args.seeds:
                rollout, result = collect(policy, adapter, seed, cuda_rng, args.max_steps, args.shaping,
                                          render_samples=metadata.get('render_samples'))
                samples.extend(rollout)
                successes += result['success']
                iteration_controls += result['steps']
                total += result['steps']
                result.update(reward_sum=sum(t['reward'] for t in rollout),
                              rewarded_chunks=sum(t['reward'] != 0 for t in rollout))
                stream.write(json.dumps(dict(iteration=iteration, **result)) + '\n')
                stream.flush()
            with torch.no_grad():
                error = max(abs(float(recompute(model, encoding, t)[0]) - t['old_logp'])
                            for t in (samples[0], samples[len(samples) // 2], samples[-1]))
            if error > 0.05:
                raise RuntimeError(f'PPO log probabilities diverged by {error:g}')
            result = dict(sft_sha256=source_hash, norm_stats_sha256=stats_hash, horizon=HORIZON, executed=EXECUTE,
                          optimizer_iterations=0, successes=successes, trials=len(args.seeds), total_controls=total,
                          seeds=args.seeds, rng_seed=args.seed, iteration_controls=iteration_controls,
                          action_chunks=len(samples), reward_min=min(t['reward'] for t in samples),
                          reward_max=max(t['reward'] for t in samples),
                          rewarded_chunks=sum(t['reward'] != 0 for t in samples),
                          shaping=args.shaping, lr=args.lr, critic_lr=args.critic_lr, weight_decay=0,
                          clip=PPO_CLIP, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                          max_steps=args.max_steps, replay_logprob_abs_error=error)
            result['render_samples'] = metadata.get('render_samples')
            result['base_sha256'] = base_hash
            result['train_action_heads'] = metadata.get('train_action_heads', True)
            result['training_script_sha256'] = file_sha256(__file__)
            result['shared_ppo_script_sha256'] = file_sha256(Path(__file__).with_name('ppo_pi05.py'))
            if args.iterations:
                if not any(t['reward'] for t in samples):
                    raise RuntimeError('The rollouts provided no reward for a PPO update')
                micro_batch = 8
                grad = update(model, encoding, optimizer, samples, rng, micro_batch=micro_batch)
                delta = actor_delta(model, original)
                if grad <= 0 or delta <= 0:
                    raise RuntimeError('PPO did not change policy parameters')
                optimizer_steps = (len(samples) + micro_batch - 1) // micro_batch
                total_optimizer_steps += optimizer_steps
                weights = trainable_snapshot(model)
                result.update(optimizer_iterations=iteration + 1, actor_params=names, actor_grad_l2=grad,
                              actor_delta_from_sft=delta, optimizer_steps=optimizer_steps,
                              total_optimizer_steps=total_optimizer_steps,
                              actor_parameter_changes=actor_change_evidence(weights, original))
                torch.save(dict(policy=weights, optimizer=optimizer.state_dict(), metadata=metadata,
                                ppo=result), args.output / f'ppo-{iteration + 1}.pt')
            result['wall_seconds'] = time.monotonic() - start
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            (args.output / f'iteration-{iteration + 1}.json').write_text(json.dumps(result, indent=2) + '\n',
                                                                        encoding='utf-8')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
