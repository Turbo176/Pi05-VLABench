"""Train a shared π0.5 checkpoint on three VLABench tasks with RLinf's Pi0."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from pi05_data import LeRobotTrajectories, TASKS
from pi05_runtime import PolicyIO, actor_delta, build_model, restore_trainable, trainable_snapshot
from vlabench_policy_io import ACTUATOR_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('train', 'manifest', 'stats', 'base', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--updates', required=True, type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--execute-steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--smoke', action='store_true', help='Sample only the first trajectory of each task')
    args = parser.parse_args()
    if args.updates < 1 or not 1 <= args.execute_steps <= 50 or not torch.cuda.is_available():
        parser.error('Expected CUDA, positive update count and valid action execution length')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = LeRobotTrajectories(args.train, args.manifest, {0, 1, 2} if args.smoke else None)
    model, cfg = build_model(args.base, 'cuda', args.execute_steps)
    policy = PolicyIO(args.stats, 'cuda', cfg.max_token_len)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output / 'sft.pt'
    metadata = {'scope': 'three-episode smoke' if args.smoke else 'multitask SFT', 'seed': args.seed,
                'stats_sha256': policy.stats_sha256, 'execute_steps': args.execute_steps,
                'base_bytes': args.base.stat().st_size, 'torch': str(torch.__version__),
                'trainable_parameters': sum(p.numel() for p in trainable), 'lr': args.lr,
                'training_episodes': {key: len(value) for key, value in data.episodes.items()}}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step in range(1, args.updates + 1):
        task = TASKS[(step - 1) % len(TASKS)]
        example, episode, frame = data.sample(rng, task)
        obs, target = policy.encode(example)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(obs, target, train=False)[..., :ACTUATOR_DIM].mean()
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
            objective = model.compute_loss(obs, target, train=False)[..., :ACTUATOR_DIM].mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            active_optimizer.step()
            return float(objective.detach())

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
