"""Record reset signatures for paired pose-policy evaluations."""
import argparse
import json
import logging
from pathlib import Path

from dm_control.rl.control import PhysicsError

from convert_lerobot import POSE_TASKS
from evaluate_vlabench_reference import build_episode
from select_fruit import open_episode, set_fruit_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=POSE_TASKS, required=True)
    parser.add_argument('--prompt-mode', choices=('official', 'paraphrase', 'alternate_target'), default='official')
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-samples', type=int, choices=(0, 2, 4, 8))
    parser.add_argument('--resume', action='store_true', help='Keep previously pinned resets after checking their identity')
    parser.add_argument('--reset-attempts', type=int, default=4)
    parser.add_argument('--invalid-resets', type=Path,
                        help='Record and skip reset PhysicsErrors after all attempts; no policy is executed')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.reset_attempts < 1:
        parser.error('Use distinct evaluation seeds and a positive reset attempt count')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and args.output.exists():
        for row in map(json.loads, args.output.read_text(encoding='utf-8').splitlines()):
            if (row['seed'] in completed or row['seed'] not in args.seeds or row['task'] != args.task
                    or row['prompt_mode'] != args.prompt_mode or row.get('render_samples') != args.render_samples):
                raise ValueError('Previously pinned cases have a different identity')
            completed.add(row['seed'])
    invalid = []
    if args.invalid_resets:
        args.invalid_resets.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and args.invalid_resets.exists():
            invalid = list(map(json.loads, args.invalid_resets.read_text(encoding='utf-8').splitlines()))
            if any(row['task'] != args.task or row['prompt_mode'] != args.prompt_mode
                   or row['render_samples'] != args.render_samples for row in invalid):
                raise ValueError('Previously excluded resets have a different identity')
        elif args.invalid_resets.exists():
            raise FileExistsError(args.invalid_resets)
    excluded = {row['seed'] for row in invalid}
    if completed & excluded:
        raise ValueError('An initialization cannot be both valid and excluded')
    with args.output.open('a' if args.resume else 'x', encoding='utf-8') as stream:
        for seed in args.seeds:
            if seed in completed or seed in excluded:
                continue
            errors = []
            for attempt in range(1, args.reset_attempts + 1):
                try:
                    with open_episode(seed, args.task, render_samples=args.render_samples) as env:
                        set_fruit_prompt(env, args.prompt_mode)
                        # The generator body is never run, so the simulator remains at reset.
                        row, rollout = build_episode(env, None, seed, 1, False,
                                                     task=args.task, prompt_mode=args.prompt_mode)
                        rollout.close()
                except PhysicsError as error:
                    errors.append(str(error))
                    logging.warning('Reset %s/%s attempt %s: %s', args.task, seed, attempt, error)
                    continue
                row['reset_attempts'] = attempt
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                completed.add(seed)
                break
            else:
                if not args.invalid_resets:
                    raise RuntimeError(f'No valid reset for {args.task}/{seed} after {args.reset_attempts} attempts')
                record = dict(task=args.task, seed=seed, prompt_mode=args.prompt_mode,
                              render_samples=args.render_samples, attempts=args.reset_attempts,
                              reason='reset_physics_error_before_policy_execution', errors=errors)
                with args.invalid_resets.open('a', encoding='utf-8') as rejected:
                    rejected.write(json.dumps(record) + '\n')
                invalid.append(record)
                excluded.add(seed)
    print(json.dumps(dict(task=args.task, prompt_mode=args.prompt_mode, valid_resets=len(completed),
                          invalid_reset_seeds=sorted(excluded))), flush=True)


if __name__ == '__main__':
    main()
