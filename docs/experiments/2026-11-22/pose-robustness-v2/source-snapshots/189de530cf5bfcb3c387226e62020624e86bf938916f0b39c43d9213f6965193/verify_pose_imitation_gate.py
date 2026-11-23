"""Require repeated autonomous policy success from one declared training reset."""

import argparse
import json
from pathlib import Path
import time

from diagnose_pose_imitation import pin_training_reset
from pi05_runtime import file_sha256
from reference_pose_lerobot import load_pose_split
from train_reference_aligned_fruit import evaluate, load_aligned_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'run', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', required=True, choices=('select_fruit', 'insert_flower'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--execute', type=int, choices=(1, 5), default=5,
                        help='Controls per prediction; a different value defines a separate diagnostic profile')
    parser.add_argument('--stage', choices=('base', 'sft'), default='sft')
    parser.add_argument('--video', action='store_true', help='Record repeat one; all repeats still execute model actions')
    parser.add_argument('--stop-on-failure', action='store_true',
                        help='A completed failure already rejects the gate; retain it and skip redundant repeats')
    args = parser.parse_args()
    if args.repeats < 3 or args.output.exists():
        parser.error('Use at least three repeats and a fresh output directory')
    selected = [ep for ep in load_pose_split(args.dataset, 'train')
                if (ep.task, ep.seed) == (args.task, args.seed)]
    if len(selected) != 1 or selected[0].policy_prefix_steps:
        parser.error('Declare a single complete training demonstration without a policy prefix')
    episode, = selected
    expected = pin_training_reset(episode, episode.render_samples)
    args.output.mkdir(parents=True)
    (args.output / 'fixed-training-reset.jsonl').write_text(
        json.dumps(expected[args.seed]) + '\n', encoding='utf-8')
    policy = load_aligned_policy(args.run, args.base, args.fork, stage=args.stage)
    started = time.monotonic()
    report = dict(kind='repeated_autonomous_imitation_on_one_seen_training_reset',
        script_sha256=file_sha256(__file__), stage=args.stage, task=args.task, seed=args.seed,
        sft_sha256=file_sha256(args.run / 'sft.pt'),
        dataset_verification_sha256=file_sha256(args.dataset / 'verification.json'),
        initial_state_sha256=episode.initial_state, requested_repeats=args.repeats,
        model_weights_loaded_once=True, environment_reset_fresh_for_every_repeat=True,
        sampler_seed_rule='Reset JAX sampler to the declared episode seed on every independent environment reset.',
        expert_actions_supplied_to_policy=False, optimizer_updates=0, max_control_steps=600,
        replan_steps=args.execute, control_hz=10,
        repeats=[], passed=False, complete=False)
    destination = args.output / 'gate.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for repeat in range(1, args.repeats + 1):
        out = args.output / f'repeat-{repeat}'
        evaluate(policy, out, expected, 600, args.video and repeat == 1, args.stage,
                 seeds=[args.seed], trace=True, execute=args.execute,
                 task=args.task, render_samples=episode.render_samples)
        row, = [json.loads(line) for line in (out / 'episodes.jsonl').read_text().splitlines()]
        if row['initial_state_sha256'] != episode.initial_state:
            raise ValueError('The repeat no longer uses the declared training reset')
        report['repeats'].append(dict(repeat=repeat, **row))
        report['wall_seconds'] = time.monotonic() - started
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(repeat=repeat, task=args.task, seed=args.seed, success=row['success'],
                              controls=row['steps'], action_sha256=row['action_sha256'])), flush=True)
        if args.stop_on_failure and not row['success']:
            report['early_stop_reason'] = 'A completed autonomous failure rejects the all-repeats-success gate.'
            break
    report['complete'] = True
    report['successes'] = sum(row['success'] for row in report['repeats'])
    report['completed_repeats'] = len(report['repeats'])
    report['passed'] = report['completed_repeats'] == args.repeats and report['successes'] == args.repeats
    report['all_full_action_hashes_identical'] = len({row['action_sha256'] for row in report['repeats']}) == 1
    report['wall_seconds'] = time.monotonic() - started
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'repeats'}), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
