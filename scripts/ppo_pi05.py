"""On-policy PPO for RLinf's π0.5 action policy on official VLABench fruit scenes."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from pi05_runtime import ACTOR_MODULES, PolicyIO, actor_delta, build_model, restore_trainable, trainable_snapshot
from select_fruit import camera_names, open_episode, robot_state, success
from vlabench_policy_io import IMAGE_SIZE, SCENE_CAMERA, WRIST_CAMERA, simulator_inputs

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2


def fruit_potential(env):
    """Reward-only geometry. It is never part of an observation."""
    physics = env.physics
    fruit = env.task.entities[env.task.target_entity].get_xpos(physics)
    plate = env.task.entities[env.task.target_container].get_xpos(physics)
    gripper = env.robot.get_end_effector_pos(physics)
    reach = np.linalg.norm(gripper - fruit)
    place = np.linalg.norm((fruit - plate)[:2])
    return -float(min(reach, 0.7) + min(place, 0.5))


def collect(model, policy, seed, max_steps, execute_steps, rng, shaping=0.0):
    transitions = []
    steps = 0
    with open_episode(seed, 'select_fruit') as env:
        if success(env):
            raise RuntimeError('The start already satisfies the goal')
        ids = {key: camera_names(env).index(key) for key in (SCENE_CAMERA, WRIST_CAMERA)}
        prompt = env.task.get_instruction()
        potential = fruit_potential(env) if shaping else 0.0
        while steps < max_steps:
            views = {key: env.render(camera_id=id_, height=IMAGE_SIZE, width=IMAGE_SIZE) for key, id_ in ids.items()}
            row = simulator_inputs(views, robot_state(env), prompt)
            with torch.no_grad():
                obs, _ = policy.encode(row)
                actions, result = model._predict_train(obs, noise=None, rng=rng, compute_values=True)
            # Retain the raw allowlisted observation. Re-encode it for actor recomputation.
            forward = result['forward_inputs']
            executed = 0
            for action in actions[0, :min(execute_steps, max_steps - steps)].cpu().numpy():
                timestep = env.step(action)
                executed += 1
                steps += 1
                won = success(env)
                if won or timestep.last():
                    break
            next_potential = fruit_potential(env) if shaping else 0.0
            reward = float(won) + shaping * (next_potential - potential)
            potential = next_potential
            transitions.append({'row': row, 'chain': forward['chains'].cpu(),
                                'denoise_inds': forward['denoise_inds'].cpu(),
                                'old_logp': float(result['prev_logprobs'][0, :executed].sum()),
                                'value': float(result['prev_values'].item()),
                                'executed': executed, 'reward': reward})
            if won or timestep.last():
                break
    # The configured time limit is the end of the finite-horizon PPO objective.
    next_value = advantage = 0.0
    for transition in reversed(transitions):
        delta = transition['reward'] + GAMMA * next_value - transition['value']
        advantage = delta + GAMMA * GAE_LAMBDA * advantage
        transition['advantage'] = advantage
        transition['return'] = advantage + transition['value']
        next_value = transition['value']
    return transitions, steps, won


def recompute(model, policy, transition):
    obs, _ = policy.encode(transition['row'])
    replay = {'chains': transition['chain'].to('cuda'), 'denoise_inds': transition['denoise_inds'].to('cuda'),
              'tokenized_prompt': obs.tokenized_prompt, 'tokenized_prompt_mask': obs.tokenized_prompt_mask,
              'obs_state': obs.state}
    replay.update({f'obs_image__{key}': image for key, image in obs.images.items()})
    replay.update({f'obs_image_mask__{key}': mask for key, mask in obs.image_masks.items()})
    result = model.default_forward(replay)
    return result['logprobs'][0, :transition['executed']].sum(), result['values'].reshape(())


def update(model, policy, optimizer, transitions, rng, micro_batch):
    advantages = np.array([t['advantage'] for t in transitions])
    advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
    actor = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith(ACTOR_MODULES)]
    critic = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith('value_head.')]
    indices = rng.permutation(len(transitions))
    max_grad = 0.0
    for offset in range(0, len(indices), micro_batch):
        batch = indices[offset:offset + micro_batch]
        optimizer.zero_grad(set_to_none=True)
        for idx in batch:
            t = transitions[idx]
            logp, value = recompute(model, policy, t)
            log_ratio = logp - t['old_logp']
            ratio = log_ratio.clamp(-20, 20).exp()
            advantage = float(advantages[idx])
            surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * advantage)
            value_loss = (value - t['return']).square()
            ((-surrogate + 0.5 * value_loss) / len(batch)).backward()
        grad = torch.nn.utils.clip_grad_norm_(actor, 1.0, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(critic, 1.0, error_if_nonfinite=True)
        max_grad = max(max_grad, float(grad))
        optimizer.step()
    return max_grad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('sft', 'base', 'stats', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--seeds', required=True, type=int, nargs='+')
    parser.add_argument('--iterations', required=True, type=int)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--minibatch', type=int, default=8)
    parser.add_argument('--shaping', type=float, default=0.0, help='Weight of fruit geometry progress in the reward')
    parser.add_argument('--seed', type=int, default=372)
    args = parser.parse_args()
    if (not torch.cuda.is_available() or args.iterations < 1 or len(set(args.seeds)) < 2
            or any(not 1201 <= seed <= 1299 for seed in args.seeds)
            or not 1 <= args.max_steps <= 600 or args.minibatch < 1 or not 0 <= args.shaping <= 1):
        parser.error('Expected CUDA, at least two distinct PPO starts in 1201–1299, and valid budgets')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoint = torch.load(args.sft, map_location='cpu', weights_only=True)
    execute_steps = checkpoint['metadata']['execute_steps']
    model, cfg = build_model(args.base, 'cuda', execute_steps, rl=True)
    restore_trainable(model, checkpoint['policy'])
    original = trainable_snapshot(model, actor_only=True)
    policy = PolicyIO(args.stats, 'cuda', cfg.max_token_len)
    if policy.stats_sha256 != checkpoint['metadata']['stats_sha256']:
        raise RuntimeError('SFT normalization hash mismatch')

    def output_transform(outputs):
        return {'actions': policy.decode(torch.as_tensor(outputs['actions'])[None])}

    model.setup_transforms([], [output_transform])
    args.output.mkdir(parents=True, exist_ok=False)
    groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(ACTOR_MODULES)],
               'lr': 3e-6},
              {'params': [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('value_head.')],
               'lr': 1e-4}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    cuda_rng = torch.Generator(device='cuda').manual_seed(args.seed)
    total_steps = 0
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(args.iterations):
        samples, wins, controls = [], 0, 0
        for seed in args.seeds:
            rollout, steps, won = collect(model, policy, seed, args.max_steps, execute_steps, cuda_rng, args.shaping)
            samples.extend(rollout)
            controls += steps
            wins += won
            print(json.dumps({'iteration': iteration, 'seed': seed, 'steps': steps, 'success': won}), flush=True)
        total_steps += controls
        with torch.no_grad():
            dev = [abs(float(recompute(model, policy, t)[0]) - t['old_logp']) for t in samples[:3]]
        if max(dev) > 0.05:
            raise RuntimeError(f'Sampling and actor log probabilities differ: {dev}')
        if not any(t['reward'] != 0 for t in samples):
            raise RuntimeError(f'No success reward in {len(args.seeds)} starts after {total_steps} environment controls')
        grad = update(model, policy, optimizer, samples, rng, args.minibatch)
        delta = actor_delta(model, original)
        if grad <= 0 or delta <= 0:
            raise RuntimeError('PPO did not update the action policy')
        metadata = {'iteration': iteration + 1, 'train_successes': wins, 'train_attempts': len(args.seeds),
                    'total_controls': total_steps, 'shaping_weight': args.shaping,
                    'mean_chunk_reward': float(np.mean([t['reward'] for t in samples])), 'max_actor_grad_l2': grad, 'actor_max_abs_delta_from_sft': delta,
                    'preupdate_logp_max_abs_error': max(dev), 'sft_sha256': hashlib.sha256(args.sft.read_bytes()).hexdigest(),
                    'stats_sha256': policy.stats_sha256, 'execute_steps': execute_steps,
                    'wall_seconds': time.monotonic() - start,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        with (args.output / 'updates.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(metadata) + '\n')
        torch.save({'policy': trainable_snapshot(model), 'optimizer': optimizer.state_dict(), 'metadata': metadata},
                   args.output / f'ppo-{iteration + 1}.pt')
        print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
