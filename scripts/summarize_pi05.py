"""Summarize matched evaluation runs from base, multitask SFT and PPO."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


Z95 = 1.959963984540054
PROTOCOL = Path(__file__).resolve().parents[1] / "configs/evaluation-v1.json"


def wilson(successes, total):
    fraction = successes / total
    scale = 1 + Z95**2 / total
    center = (fraction + Z95**2 / (2 * total)) / scale
    half_width = Z95 * math.sqrt(fraction * (1 - fraction) / total + Z95**2 / (4 * total**2)) / scale
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_case = {(row['task'], row['seed'], row['prompt_mode']): row for row in rows}
    if not rows or len(by_case) != len(rows):
        raise ValueError(f'Empty or duplicate evaluation cases in {path}')
    return by_case


def summarize(rows):
    wins = sum(row['success'] for row in rows)
    total = len(rows)
    return {'successes': wins, 'attempts': total, 'success_rate': wins / total,
            'wilson_95': wilson(wins, total),
            'failure_reasons': dict(Counter(row['stop_reason'] for row in rows if not row['success'])),
            'failure_types': dict(Counter(row['failure_type'] for row in rows if not row['success'])),
            'environment_control_steps': sum(row['steps'] for row in rows),
            'rollout_wall_seconds': sum(row['wall_seconds'] for row in rows)}


def initial_rgb_key(first, second):
    key = 'initial_policy_rgb_sha256'
    return key if key in first and key in second else 'initial_rgb_sha256'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pose', action='store_true', help='Read pose-policy records')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL,
                        help='Frozen evaluation protocol used to check a complete suite')
    parser.add_argument('--suite', choices=('ordinary', 'fruit_language'),
                        help='Require all preregistered cases and all three policy stages')
    args = parser.parse_args()
    reference = load_rows(args.runs[0])
    protocol_keys = ('suite', 'instruction', 'noise_seed', 'execute_steps', 'max_control_steps',
                     'initial_state_sha256', 'stats_sha256')
    if args.pose:
        protocol_keys = ('instruction', 'replan_steps', 'max_control_steps', 'control_hz', 'sampler',
                         'initial_state_sha256', 'norm_stats_sha256', 'reference_model_revision')
    if args.suite:
        protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
        suite = protocol[args.suite]
        low, high = suite['seeds_inclusive']
        task_seeds = suite.get('seeds_by_task', {})
        if set(task_seeds) - set(suite['tasks']):
            raise ValueError('Protocol seed overrides name an unknown task')
        expected = {(task, seed, mode) for task in suite['tasks']
                    for seed in task_seeds.get(task, range(low, high + 1))
                    for mode in suite.get('prompt_modes', ['official'])}
        missing = sorted(expected - reference.keys())
        extra = sorted(reference.keys() - expected)
        if missing or extra:
            raise ValueError(f'{args.suite}: missing cases {missing}, extra cases {extra}')
        if any((not args.pose and row['suite'] != args.suite)
               or row['max_control_steps'] != protocol['max_control_steps']
               for row in reference.values()):
            raise ValueError(f'{args.suite}: incorrect suite or control budget')
        for key in ('render_samples', 'control_hz', 'replan_steps'):
            if args.pose and key in protocol and any(row.get(key) != protocol[key] for row in reference.values()):
                raise ValueError(f'{args.suite}: incorrect {key}')
    groups = sorted({(task, prompt) for task, _, prompt in reference})
    reports = {}
    indexed = {}
    for path in args.runs:
        run = load_rows(path)
        if run.keys() != reference.keys():
            raise ValueError(f'Case set differs from the reference in {path}')
        stage = next(iter(run.values()))['stage']
        if stage in reports or any(row['stage'] != stage for row in run.values()):
            raise ValueError(f'Inconsistent or duplicate stage {stage}')
        for case, row in run.items():
            if row.get('action_space', 'absolute') != reference[case].get('action_space', 'absolute'):
                raise ValueError(f'Action space differs for {case} in {path}')
            for key in (*protocol_keys, initial_rgb_key(row, reference[case])):
                if row[key] != reference[case][key]:
                    raise ValueError(f'{key} differs for {case} in {path}')
            if args.pose:
                for key in ('base_sha256', 'render_samples'):
                    if row.get(key) != reference[case].get(key):
                        raise ValueError(f'{key} differs for {case} in {path}')
        indexed[stage] = run
        reports[stage] = {f'{task}/{prompt}': summarize([row for case, row in run.items()
                                                        if (case[0], case[2]) == (task, prompt)])
                          for task, prompt in groups}
    if args.suite and indexed.keys() != {'base', 'sft', 'ppo'}:
        raise ValueError('Full evaluation requires base, SFT, and PPO results')
    if args.pose and 'sft' in indexed and 'ppo' in indexed:
        for case, row in indexed['sft'].items():
            if row['sft_sha256'] != indexed['ppo'][case]['sft_sha256']:
                raise ValueError(f'The PPO parent differs from the evaluated SFT checkpoint for {case}')
    pairs = {}
    for earlier, later in [('base', 'sft'), ('sft', 'ppo'), ('base', 'ppo')]:
        if earlier in indexed and later in indexed:
            groups_report = {}
            for task, prompt in groups:
                changes = Counter((indexed[earlier][k]['success'], indexed[later][k]['success'])
                                  for k in reference if (k[0], k[2]) == (task, prompt))
                groups_report[f'{task}/{prompt}'] = {
                    'fail_to_success': changes[False, True], 'success_to_fail': changes[True, False],
                    'both_failed': changes[False, False], 'both_succeeded': changes[True, True],
                }
            pairs[f'{earlier}_to_{later}'] = groups_report
    result = {'cases_per_stage': len(reference), 'stages': reports, 'paired_outcomes': pairs}
    if args.suite == 'fruit_language':
        language_reset_keys = ('initial_state_sha256',) if args.pose else ('initial_state_sha256', 'noise_seed')
        action_key = 'action_sha256' if args.pose else 'action_chunks_sha256'
        language_effects = {}
        for stage, run in indexed.items():
            language_effects[stage] = {}
            for mode in ('paraphrase', 'alternate_target'):
                modified = [(case, row) for case, row in run.items() if case[2] == mode]
                changed = changed_first = 0
                for (task, seed, _), row in modified:
                    original = run[(task, seed, 'official')]
                    for key in (*language_reset_keys, initial_rgb_key(row, original)):
                        if row[key] != original[key]:
                            raise ValueError(f'{stage} {task} {seed}: {key} differs across prompts ({mode})')
                    changed += row[action_key] != original[action_key]
                    if args.pose:
                        changed_first += row['first_action_sha256'] != original['first_action_sha256']
                effects = {'changed_action_traces': changed, 'matched_seeds': len(modified)}
                if args.pose:
                    effects['changed_first_commands'] = changed_first
                language_effects[stage][mode] = effects
        result['language_effects'] = language_effects
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
