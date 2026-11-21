"""Evaluate a π0.5 checkpoint through the official VLABench control loop."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from threading import Lock
import time

import numpy as np
import torch

from evaluate_policy import ROOT, run_episode
from pi05_data import TASKS
from pi05_runtime import DENOISE_STEPS, PolicyIO, build_model, restore_trainable
from select_fruit import digest_arrays, open_episode, physics_state, render_views, set_fruit_prompt, video


RESET_LOCK = Lock()
INFERENCE_LOCK = Lock()


def load_stage(args):
    model, config = build_model(args.base, 'cuda', args.execute_steps, rl=args.stage in ('handoff', 'ppo'))
    policy = PolicyIO(args.stats, 'cuda', config.max_token_len)
    if args.stage == 'base':
        if args.checkpoint:
            raise ValueError('The base stage accepts only the converted official weights')
        return model, policy, None
    if not args.checkpoint:
        raise ValueError('This stage needs a checkpoint')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    if metadata['stats_sha256'] != policy.stats_sha256 or metadata['execute_steps'] != args.execute_steps:
        raise ValueError('Checkpoint preprocessing or execution budget differs from this run')
    restore_trainable(model, checkpoint['policy'])
    return model, policy, hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()


def evaluate(model, policy, task, seed, prompt_mode, execute_steps, max_steps, noise_seed, video_path=None, label=None):
    start = time.monotonic()
    action_digest = hashlib.sha256()
    rng = torch.Generator(device='cuda').manual_seed(noise_seed)
    # VLABench seeds the process RNGs and mutates task defaults at reset.
    with RESET_LOCK:
        env = open_episode(seed, task)
    with env:
        if abs(env.control_timestep() - 0.1) > 1e-8:
            raise RuntimeError('Expected 10 Hz control')
        set_fruit_prompt(env, prompt_mode)
        prompt = env.task.get_instruction()
        initial_rgb = render_views(env)
        target = env.task.entities[env.task.target_entity]
        initial_target = target.get_xpos(env.physics).copy()
        furthest_target_displacement = 0.0
        fridge = env.task.entities[env.task.init_container] if task == 'take_out_cool_drink' else None
        fridge_opened = False
        hashes = {'initial_state_sha256': digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat),
                  'initial_rgb_sha256': digest_arrays(initial_rgb)}

        @torch.inference_mode()
        def infer(row):
            with INFERENCE_LOCK:
                obs, _ = policy.encode(row)
                actions = policy.decode(model.sample_actions(obs, num_steps=DENOISE_STEPS, rng=rng))
            action_digest.update(actions.tobytes())
            return actions

        frames = [initial_rgb] if video_path else None

        def record(current_env):
            nonlocal fridge_opened, furthest_target_displacement
            physics = current_env.physics
            distance = np.linalg.norm(target.get_xpos(physics) - initial_target)
            furthest_target_displacement = max(furthest_target_displacement, float(distance))
            if fridge and not fridge_opened:
                fridge_opened = fridge.is_open(physics)
            if frames is not None:
                frames.append(render_views(current_env))

        result = run_episode(env, infer, prompt, max_steps=max_steps,
                             execute_steps=execute_steps, initial_rgb=initial_rgb, after_step=record)
        fridge_closed = fridge.is_closed(env.physics) if fridge else None
    failure_type = None
    if not result['success']:
        if fridge_closed is not None and not fridge_opened:
            failure_type = 'fridge_never_reached_open_threshold'
        elif furthest_target_displacement < 0.03:
            failure_type = 'target_displacement_under_3cm'
        elif fridge_closed is False:
            failure_type = 'target_moved_fridge_not_closed'
        else:
            failure_type = 'target_moved_goal_unmet'
    if video_path:
        video(video_path, frames, 10, label, prompt, crf=18)
    return {'task': task, 'seed': seed, 'prompt_mode': prompt_mode, 'instruction': prompt,
            'noise_seed': noise_seed, 'failure_type': failure_type, 'target_max_displacement_m': furthest_target_displacement,
            'fridge_opened': fridge_opened if fridge_closed is not None else None, 'fridge_closed': fridge_closed,
            'action_chunks_sha256': action_digest.hexdigest(), 'wall_seconds': time.monotonic() - start, **hashes, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'stats', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--stage', choices=('base', 'sft', 'handoff', 'ppo'), required=True)
    parser.add_argument('--suite', choices=('smoke', 'development', 'ppo_development', 'ordinary', 'fruit_language'), required=True)
    parser.add_argument('--execute-steps', type=int, default=10)
    parser.add_argument('--max-seeds-per-task', type=int)
    parser.add_argument('--case', nargs=3, metavar=('TASK', 'SEED', 'PROMPT'),
                        help='Reproduce one case in a separate log within the specified suite')
    parser.add_argument('--workers', default=1, type=int, help='Concurrent MuJoCo rollouts per loaded policy')
    parser.add_argument('--resume', action='store_true', help='Append cases after checking the recorded run identity')
    parser.add_argument('--video-dir', type=Path, help='Record the first ordinary test seed per task')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    protocol = json.loads((ROOT / 'configs/evaluation-v1.json').read_text(encoding='utf-8'))
    if args.suite == 'smoke':
        cases = [(task, 42 if task != 'take_out_cool_drink' else 43, 'official') for task in TASKS]
        max_steps = 20
    elif args.suite == 'development':
        cases = [(TASKS[0], 43, 'official'), (TASKS[1], 46, 'official'), (TASKS[2], 46, 'official')]
        max_steps = protocol['max_control_steps']
    elif args.suite == 'ppo_development':
        cases = [(TASKS[0], seed, 'official') for seed in range(1201, 1205)]
        max_steps = protocol['max_control_steps']
    else:
        suite = protocol[args.suite]
        low, high = suite['seeds_inclusive']
        if args.max_seeds_per_task is not None:
            if args.max_seeds_per_task < 1:
                parser.error('--max-seeds-per-task must be positive')
            high = min(high, low + args.max_seeds_per_task - 1)
        cases = [(task, seed, prompt) for task in suite['tasks'] for seed in range(low, high + 1)
                 for prompt in suite.get('prompt_modes', ['official'])]
        max_steps = protocol['max_control_steps']
    if args.case:
        task, seed, prompt = args.case
        try:
            case = task, int(seed), prompt
        except ValueError:
            parser.error('--case SEED must be an integer')
        if case not in cases:
            parser.error('--case is outside the selected suite')
        cases = [case]
    if args.suite == 'fruit_language':
        language_cases = json.loads((ROOT / 'configs/evaluation-fruit-prompts-v1.json').read_text(encoding='utf-8'))
        expected_instructions = {case['seed']: case['prompts'] for case in language_cases['cases']}
    else:
        expected_instructions = {}
    model, policy, checkpoint_sha256 = load_stage(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(stage=args.stage, suite=args.suite, checkpoint_sha256=checkpoint_sha256,
                    execute_steps=args.execute_steps, stats_sha256=policy.stats_sha256, max_control_steps=max_steps)
    completed = set()
    if args.resume and args.output.is_file():
        for line in args.output.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = row['task'], row['seed'], row['prompt_mode']
            if key in completed or key not in cases or any(row[name] != value for name, value in identity.items()):
                raise ValueError(f'Inconsistent saved result for {key} in {args.output}')
            completed.add(key)
    def run_case(case):
        task, seed, prompt = case
        video_path = None
        if args.video_dir and args.suite == 'ordinary' and seed == low:
            video_path = args.video_dir / f'{args.stage}-{task}-{seed}.mp4'
        label = f'Pi0.5 {args.stage} | {task} | seed {seed}'
        result = evaluate(model, policy, task, seed, prompt, args.execute_steps, max_steps,
                          noise_seed=seed, video_path=video_path, label=label)
        if expected_instructions and result['instruction'] != expected_instructions[seed][prompt]['instruction']:
            raise RuntimeError(f'Frozen prompt differs in case {seed} {prompt}')
        result.update(identity)
        return result

    def write(log, result):
        log.write(json.dumps(result) + '\n')
        log.flush()
        print(f'{args.stage} {result["task"]} {result["seed"]} {result["prompt_mode"]} '
              f'{result["stop_reason"]} {result["steps"]}', flush=True)

    pending = [case for case in cases if case not in completed]
    with args.output.open('a' if args.resume else 'x', encoding='utf-8') as log:
        if args.workers == 1:
            for case in pending:
                write(log, run_case(case))
        else:
            # Keep different task types out of the same batch. VLABench stores task defaults globally.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for _, group in groupby(pending, itemgetter(0)):
                    futures = [pool.submit(run_case, case) for case in group]
                    try:
                        for future in as_completed(futures):
                            write(log, future.result())
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise


if __name__ == '__main__':
    main()
