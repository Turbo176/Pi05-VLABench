"""Evaluate one shared pose checkpoint on the registered development cases."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
from threading import Lock

from convert_lerobot import POSE_TASKS
import train_reference_aligned_fruit as trainer
from train_reference_aligned_fruit import evaluate, load_aligned_policy, paired_episodes


class LockedPolicy:
    """Keep each episode's sampler state separate while sharing one GPU model."""

    def __init__(self, policy, lock):
        self.policy = copy.copy(policy)
        self.policy.provenance = dict(policy.provenance)
        self.lock = lock

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def infer(self, observation):
        with self.lock:
            return self.policy.infer(observation)


def verify_serial(rows, path):
    reference = {(r['task'], r['seed'], r['prompt_mode']): r
                 for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}
    keys = ('success', 'steps', 'action_sha256', 'first_action_sha256',
            'ik_nonconverged_steps', 'initial_state_sha256', 'initial_policy_rgb_sha256',
            'instruction', 'render_samples', 'base_sha256', 'norm_stats_sha256', 'sampler')
    for row in rows:
        expected = reference[row['task'], row['seed'], row['prompt_mode']]
        for key in keys:
            if row[key] != expected[key]:
                raise ValueError(f"Serial comparison differs for {row['task']}:{row['seed']} in {key}")
    return dict(serial_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                identical_cases=len(rows), compared_fields=list(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'run', 'cases', 'protocol', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--tasks', choices=POSE_TASKS, nargs='+')
    parser.add_argument('--suite', choices=('development', 'ordinary', 'fruit_language'), default='development')
    parser.add_argument('--workers', type=int, default=1,
                        help='Overlap simulator work; GPU inference and official resets remain locked')
    parser.add_argument('--verify-against', type=Path, help='Require exact agreement with existing serial rollouts')
    parser.add_argument('--stage', choices=('base', 'sft', 'ppo'), default='sft')
    parser.add_argument('--ppo-weights', type=Path)
    args = parser.parse_args()
    if (args.workers not in range(1, 5) or (args.tasks and len(args.tasks) != len(set(args.tasks)))
            or (args.stage == 'ppo') != bool(args.ppo_weights)):
        parser.error('Use distinct tasks, and PPO weights exactly when evaluating PPO')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    suite = protocol[args.suite]
    allowed_tasks = list(suite) if args.suite == 'development' else suite['tasks']
    tasks = args.tasks or allowed_tasks
    if set(tasks) - set(allowed_tasks):
        parser.error('Requested tasks must belong to the selected protocol suite')
    manifests = {}
    for task in tasks:
        modes = ['official'] if args.suite == 'development' else suite.get('prompt_modes', ['official'])
        seeds = (suite[task] if args.suite == 'development' else
                 suite.get('seeds_by_task', {}).get(task, range(suite['seeds_inclusive'][0],
                                                               suite['seeds_inclusive'][1] + 1)))
        for mode in modes:
            filename = (task if args.suite == 'development' else
                        f'{task}-language-official' if args.suite == 'fruit_language' and mode == 'official'
                        else f'{task}-{mode}')
            cases = paired_episodes(args.cases / f'{filename}.jsonl')
            if (set(cases) != set(seeds)
                    or any(row['task'] != task or row['prompt_mode'] != mode
                           or row['render_samples'] != protocol['render_samples'] for row in cases.values())):
                raise ValueError(f'Unexpected {args.suite} cases for {task}/{mode}')
            manifests[task, mode] = cases
    policy = load_aligned_policy(args.run, args.base, args.fork, args.stage, args.ppo_weights)
    if policy.provenance.get('training_scope'):
        raise ValueError('Single-episode diagnostics are not multitask development candidates')
    if policy.training_render_samples != protocol['render_samples']:
        raise ValueError('Checkpoint and protocol have different rendering profiles')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'completed.json').exists():
        raise FileExistsError('The evaluation is already complete')
    results = []
    if args.workers == 1:
        for (task, mode), cases in manifests.items():
            destination = args.output / (task if args.suite == 'development' else f'{task}-{mode}')
            evaluate(policy, destination, cases, protocol['max_control_steps'], False, args.stage,
                     seeds=sorted(cases), task=task, prompt_mode=mode, resume=True,
                     render_samples=protocol['render_samples'])
            rows = [json.loads(line) for line in (destination / 'episodes.jsonl').read_text().splitlines()]
            if len(rows) != len(cases):
                raise ValueError(f'Incomplete evaluation for {task}/{mode}')
            results.extend(rows)
    else:
        import jax

        jax.devices('cpu')  # Initialize the shared JAX runtime before starting threads.
        reset_lock, inference_lock = Lock(), Lock()
        original_open_episode = trainer.open_episode

        def locked_open_episode(*args, **kwargs):
            # Official resets seed process-global random streams and mutate task defaults.
            with reset_lock:
                return original_open_episode(*args, **kwargs)

        def run_case(task, mode, seed, cases):
            episode_policy = LockedPolicy(policy, inference_lock)
            destination = args.output / 'cases' / f'{task}-{seed}-{mode}'
            evaluate(episode_policy, destination, cases, protocol['max_control_steps'], False, args.stage,
                     seeds=[seed], task=task, prompt_mode=mode, resume=True,
                     render_samples=protocol['render_samples'])
            row, = map(json.loads, (destination / 'episodes.jsonl').read_text().splitlines())
            return row

        trainer.open_episode = locked_open_episode
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_case, task, mode, seed, cases)
                           for (task, mode), cases in manifests.items() for seed in sorted(cases)]
                for future in as_completed(futures):
                    results.append(future.result())
        finally:
            trainer.open_episode = original_open_episode
        results.sort(key=lambda r: (tasks.index(r['task']), r['prompt_mode'], r['seed']))
    destination = args.output / 'episodes.jsonl'
    destination.write_text(''.join(json.dumps(row) + '\n' for row in results), encoding='utf-8')
    report = dict(stage=args.stage, **policy.provenance,
                  protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  suite=args.suite, workers=args.workers, tasks=tasks, cases=len(results),
                  successes=sum(row['success'] for row in results))
    if args.verify_against:
        report['serial_equivalence'] = verify_serial(results, args.verify_against)
    with (args.output / 'completed.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
