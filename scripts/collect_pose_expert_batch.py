"""Screen new official expert starts and admit only fresh successful pose replays."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from dm_control.rl.control import PhysicsError

from collect_reference_fruit_pose_expert import collect
from convert_lerobot import POSE_TASKS, read_pose_record
from rerender_pose_dataset import replay
from select_fruit import open_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', choices=POSE_TASKS, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--max-admitted-per-task', type=int, default=2)
    parser.add_argument('--instruction-contains', help='Screen natural seeded instructions before executing experts')
    parser.add_argument('--stop-at-cap', action='store_true', help='Stop each task after enough verified replays')
    parser.add_argument('--render-samples', type=int, choices=(0, 2, 4, 8), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds)
            or args.max_admitted_per_task < 1):
        parser.error('Use distinct tasks/seeds and a positive admission cap')
    fruit_only = set(args.tasks) == {'select_fruit'}
    if any(not (0 <= seed < 1000 or fruit_only and 2050 <= seed < 2120) for seed in args.seeds):
        parser.error('Use training seeds below 1000, or fruit-only expansion seeds 2050-2119')
    args.output.mkdir(parents=True, exist_ok=False)
    source, replays = args.output / 'source', args.output / 'replays'
    replays.mkdir()
    admitted, counts = Counter(), Counter()
    started = time.monotonic()
    with (args.output / 'attempts.jsonl').open('x', encoding='utf-8') as stream:
        for task in args.tasks:
            for seed in args.seeds:
                if args.stop_at_cap and admitted[task] >= args.max_admitted_per_task:
                    break
                row = dict(task=task, seed=seed, render_samples=args.render_samples,
                           probe_success=False, captured_success=None, replay_success=None,
                           admitted=False)
                phase = 'expert_probe'
                try:
                    matches = True
                    if args.instruction_contains:
                        phase = 'instruction_screen'
                        with open_episode(seed, task, render_samples=args.render_samples) as env:
                            instruction = env.task.get_instruction()
                        matches = args.instruction_contains.casefold() in instruction.casefold()
                        row.update(instruction=instruction, instruction_filter_match=matches)
                    phase = 'expert_probe'
                    if matches:
                        probe = collect(seed, None, 600, task, render_samples=args.render_samples)
                        row.update(probe_success=probe['success'], probe=probe)
                    if row['probe_success'] and admitted[task] < args.max_admitted_per_task:
                        phase = 'expert_capture'
                        capture = collect(seed, source, 600, task, render_samples=args.render_samples)
                        row.update(captured_success=capture['success'], capture=capture)
                        if capture['success']:
                            name = str(seed) if task == 'select_fruit' else f'{task}-{seed}'
                            record = read_pose_record(source / f'{name}.h5', 'train')
                            phase = 'fresh_action_replay'
                            outcome = replay(record, replays, args.render_samples)
                            row.update(replay_success=outcome['success'], replay=outcome,
                                       admitted=outcome['success'])
                except PhysicsError as error:
                    row.update(physics_error=str(error), failure_phase=phase)
                admitted[task] += int(row['admitted'])
                counts[task] += 1
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps(row), flush=True)
    summary = dict(kind='official_expert_screening_and_fresh_action_replay_not_policy_evaluation',
                   attempts=dict(counts), admitted=dict(admitted), render_samples=args.render_samples,
                   max_controls_per_attempt=600, wall_seconds=time.monotonic() - started)
    summary.update(instruction_filter=args.instruction_contains, stop_at_cap=args.stop_at_cap)
    summary['training_seed_scope'] = 'below 1000; fruit-only expansion also permits 2050-2119'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
