"""Train a shared π0.5 checkpoint on three VLABench tasks with RLinf's Pi0."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from pi05_data import LeRobotTrajectories, TASKS
from pi05_runtime import PolicyIO, actor_delta, build_model, restore_trainable, trainable_snapshot
from vlabench_policy_io import ACTION_HORIZON, ACTION_SPACES, ACTUATOR_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('train', 'manifest', 'stats', 'base', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--updates', required=True, type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--execute-steps', type=int, default=10)
    parser.add_argument('--action-space', choices=ACTION_SPACES, default='absolute')
    parser.add_argument('--lora-rank', type=int, default=0)
    parser.add_argument('--vlm-lora-rank', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--init-sft', type=Path, help='Start from trained actor weights with a fresh optimizer')
    parser.add_argument('--fruit-extra-export', type=Path, help='Verified LeRobot v2.1 export of additional fruit training scenes')
    parser.add_argument('--fruit-recovery-export', type=Path, help='Verified LeRobot v2.1 export of expert recoveries')
    parser.add_argument('--fruit-recovery-probability', type=float, default=0.35)
    parser.add_argument('--near-term-weight', type=float, default=1.0,
                        help='Relative loss weight for controls executed before replanning')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--smoke', action='store_true', help='Sample only the first trajectory of each task')
    scope.add_argument('--diagnostic-episode', type=int, help='Overfit one training episode to diagnose closed-loop control')
    parser.add_argument('--supervise-padding', action='store_true',
                        help='Include the zero-padded coordinates in the flow-matching loss')
    args = parser.parse_args()
    if args.updates < 1 or not 1 <= args.execute_steps <= 50 or not torch.cuda.is_available():
        parser.error('Expected CUDA, positive update count and valid action execution length')
    if not math.isfinite(args.near_term_weight) or args.near_term_weight <= 0:
        parser.error('--near-term-weight must be finite and positive')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    episode_filter = {0, 1, 2} if args.smoke else None
    if args.diagnostic_episode is not None:
        episode_filter = {args.diagnostic_episode}
    data = LeRobotTrajectories(args.train, args.manifest, episode_filter)
    extra = None
    if args.fruit_extra_export:
        if episode_filter is not None:
            parser.error('Additional fruit scenes require full multitask training')
        extra_manifest = args.fruit_extra_export / 'verification.json'
        original = json.loads(args.manifest.read_text(encoding='utf-8'))['splits']['train']['episodes_detail']
        extra_rows = json.loads(extra_manifest.read_text(encoding='utf-8'))['splits']['train']['episodes_detail']
        fruit_seeds = {row['seed'] for row in original if row['task'] == 'select_fruit'}
        initial_states = {row['initial_state_sha256'] for row in original}
        evaluation = json.loads((Path(__file__).resolve().parents[1] / 'configs/evaluation-v1.json').read_text())
        reserved_seeds = {43, *range(1201, 1205), *range(1211, 1215)}
        for split in ('ordinary', 'fruit_language'):
            start_seed, end_seed = evaluation[split]['seeds_inclusive']
            reserved_seeds.update(range(start_seed, end_seed + 1))
        if any(row['task'] != 'select_fruit' or row['seed'] in fruit_seeds | reserved_seeds
               or row['initial_state_sha256'] in initial_states for row in extra_rows):
            raise ValueError('Expected new fruit scenes outside the original training and evaluation starts')
        extra = LeRobotTrajectories(args.fruit_extra_export / 'train', extra_manifest,
                                    required_tasks=('select_fruit',))
        count_extra = len(extra.episodes['select_fruit'])
        extra_share = count_extra / (count_extra + len(data.episodes['select_fruit']))
    recovery = None
    if args.fruit_recovery_export:
        if episode_filter is not None or not 0 < args.fruit_recovery_probability < 1:
            parser.error('Recovery sampling requires full multitask training and probability between zero and one')
        recovery_manifest = args.fruit_recovery_export / 'verification.json'
        original = json.loads(args.manifest.read_text(encoding='utf-8'))['splits']['train']['episodes_detail']
        recoveries = json.loads(recovery_manifest.read_text(encoding='utf-8'))['splits']['train']['episodes_detail']
        fruit_seeds = {row['seed'] for row in original if row['task'] == 'select_fruit'}
        if any(row['task'] != 'select_fruit' or row['seed'] not in fruit_seeds for row in recoveries):
            raise ValueError('Expected fruit recoveries from the original training scenes only')
        recovery = LeRobotTrajectories(args.fruit_recovery_export / 'train', recovery_manifest,
                                      required_tasks=('select_fruit',))
    active_tasks = tuple(task for task in TASKS if data.episodes[task])
    if args.smoke and active_tasks != TASKS:
        raise RuntimeError('The smoke training fixture must cover all three tasks')
    run_scope = 'three-episode smoke' if args.smoke else 'multitask SFT'
    if args.diagnostic_episode is not None:
        run_scope = 'single training episode diagnostic'
    model, cfg = build_model(args.base, 'cuda', args.execute_steps,
                            lora_rank=args.lora_rank, vlm_lora_rank=args.vlm_lora_rank)
    policy = PolicyIO(args.stats, 'cuda', cfg.max_token_len, args.action_space)
    initialization = None
    if args.init_sft:
        checkpoint = torch.load(args.init_sft, map_location='cpu', weights_only=True)
        previous = checkpoint['metadata']
        expected = {'lora_rank': args.lora_rank, 'execute_steps': args.execute_steps,
                    'stats_sha256': policy.stats_sha256, 'base_bytes': args.base.stat().st_size}
        if previous.get('action_space', 'absolute') != args.action_space or any(
                previous.get(key) != value for key, value in expected.items()):
            raise ValueError('Initial SFT checkpoint uses a different model or policy adapter')
        prior_vlm_rank = previous.get('vlm_lora_rank', 0)
        prior_weights = checkpoint['policy']
        if prior_vlm_rank != args.vlm_lora_rank:
            if prior_vlm_rank != 0 or not args.vlm_lora_rank:
                raise ValueError('Warm-start rank differs from the existing VLM adapters')
            new_adapters = {name: weight for name, weight in trainable_snapshot(model).items()
                            if '.q_proj.0.' in name or '.v_proj.0.' in name}
            if not new_adapters or new_adapters.keys() & prior_weights.keys():
                raise ValueError('Expected only newly added VLM adapters')
            prior_weights = {**prior_weights, **new_adapters}
        restore_trainable(model, prior_weights)
        initialization = hashlib.sha256(args.init_sft.read_bytes()).hexdigest()
        del checkpoint
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output / 'sft.pt'
    metadata = {'scope': run_scope, 'seed': args.seed,
                'stats_sha256': policy.stats_sha256, 'execute_steps': args.execute_steps, 'lora_rank': args.lora_rank, 'action_space': args.action_space,
                'base_bytes': args.base.stat().st_size, 'torch': str(torch.__version__),
                'trainable_parameters': sum(p.numel() for p in trainable), 'lr': args.lr,
                'training_episodes': {key: len(value) for key, value in data.episodes.items()}}
    if extra is not None:
        metadata['fruit_extra'] = {'episodes': count_extra,
                                  'manifest_sha256': hashlib.sha256(extra_manifest.read_bytes()).hexdigest()}
    if args.diagnostic_episode is not None:
        metadata['diagnostic_episode'] = args.diagnostic_episode
    if recovery is not None:
        metadata['fruit_recovery'] = {'episodes': len(recoveries), 'probability': args.fruit_recovery_probability,
                                      'manifest_sha256': hashlib.sha256(recovery_manifest.read_bytes()).hexdigest()}
    metadata['near_term_weight'] = args.near_term_weight
    metadata['supervise_padding'] = args.supervise_padding
    metadata['vlm_lora_rank'] = args.vlm_lora_rank
    if initialization:
        metadata['init_sft_sha256'] = initialization
    weights_by_step = torch.ones(ACTION_HORIZON, device='cuda')
    weights_by_step[:args.execute_steps] = args.near_term_weight
    weights_by_step /= weights_by_step.mean()

    def objective(obs, target):
        per_control = model.compute_loss(obs, target, train=False)
        if not args.supervise_padding:
            per_control = per_control[..., :ACTUATOR_DIM]
        if args.near_term_weight != 1.0:
            per_control = per_control * weights_by_step[None, :, None]
        return per_control.mean()

    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step in range(1, args.updates + 1):
        task = active_tasks[(step - 1) % len(active_tasks)]
        use_recovery = (recovery is not None and task == 'select_fruit'
                        and rng.random() < args.fruit_recovery_probability)
        source = recovery if use_recovery else data
        source_name = 'expert_recovery' if use_recovery else 'original'
        if not use_recovery and extra is not None and task == 'select_fruit' and rng.random() < extra_share:
            source, source_name = extra, 'extra_expert'
        example, episode, frame = source.sample(rng, task)
        obs, target = policy.encode(example)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(obs, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite SFT loss at update {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        prior = trainable_snapshot(model, actor_only=True) if step == 1 else None
        optimizer.step()
        if step == 1:
            delta = actor_delta(model, prior)
            if grad <= 0 or delta <= 0:
                raise RuntimeError('Expected gradients and a policy update')
            metadata['first_actor_max_abs_delta'] = delta
        record = {'update': step, 'task': task, 'episode': episode, 'frame': frame,
                  'loss': float(loss.detach()), 'unclipped_grad_l2': float(grad)}
        if recovery is not None or extra is not None:
            record['source'] = source_name
        with (args.output / 'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        if step == 1 or step == args.updates or step % 20 == 0:
            print(json.dumps(record), flush=True)
    metadata.update(updates=args.updates, wall_seconds=time.monotonic() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
    weights = trainable_snapshot(model)
    optimizer_state = optimizer.state_dict()
    torch.save({'policy': weights, 'optimizer': optimizer_state,
                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'sampler_rng': rng.bit_generator.state, 'metadata': metadata}, out)
    restored = torch.load(out, map_location='cpu', weights_only=True)
    with torch.no_grad():
        trainable[0].zero_()
    restore_trainable(model, restored['policy'])
    restored_optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    restored_optimizer.load_state_dict(restored['optimizer'])
    if not restored_optimizer.state or actor_delta(model, weights) != 0:
        raise RuntimeError('Checkpoint read-back failed')
    metadata['checkpoint_readback'] = True
    if args.smoke:
        # Replay the same next optimizer step after restoring the initial RNG states.
        def next_update(active_optimizer):
            torch.set_rng_state(restored['cpu_rng'])
            torch.cuda.set_rng_state(restored['cuda_rng'])
            active_optimizer.zero_grad(set_to_none=True)
            loss = objective(obs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            active_optimizer.step()
            return float(loss.detach())

        expected_loss = next_update(optimizer)
        expected_weights = trainable_snapshot(model)
        restore_trainable(model, weights)
        actual_loss = next_update(restored_optimizer)
        mismatch = actor_delta(model, expected_weights)
        if not np.isclose(expected_loss, actual_loss, atol=1e-7, rtol=0) or mismatch > 1e-7:
            raise RuntimeError('Resumed optimizer step differs from the uninterrupted step')
        metadata.update(resume_loss=actual_loss, resume_max_abs_actor_difference=mismatch)
    (args.output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
