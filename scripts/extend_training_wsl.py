"""Screen, replay, and export additional official expert training episodes in WSL."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from convert_lerobot import read_record
from report_expansion_01 import screen_result


TASKS = ('select_fruit', 'insert_flower', 'take_out_cool_drink')
ROOT = Path(__file__).resolve().parent.parent


def run(script, *args):
    subprocess.run(['bash', f'scripts/{script}', *map(str, args)], cwd=ROOT, check=True)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def extend(args):
    tracks = {task: [] for task in TASKS}
    carry = read_json(args.previous)['splits']['train']
    if not carry['all_frames_verified']:
        raise ValueError('The base training export must pass frame readback')
    evaluation = read_json(ROOT / 'configs/evaluation-v1.json')
    tasks_to_screen = set(args.task or TASKS)
    for group in (evaluation['ordinary'], evaluation['fruit_language']):
        first, last = group['seeds_inclusive']
        if tasks_to_screen.intersection(group['tasks']) and args.first_seed <= last and args.last_seed >= first:
            raise ValueError('The training candidate range overlaps reserved evaluation cases')
        for row in carry['episodes_detail']:
            if row['task'] in group['tasks'] and first <= row['seed'] <= last:
                raise ValueError('The base export contains a reserved evaluation case')
    held_out = (read_json(Path('docs/stage2-three-task-validation-v21.json'))['splits']['validation']['episodes_detail']
                + read_json(Path('docs/stage2-lerobot-v21.json'))['splits']['test']['episodes_detail'])
    prior = carry['episodes_detail'] + held_out
    prior_identities = {(row['task'], row['seed']) for row in prior}
    prior_hashes = {row['initial_state_sha256'] for row in prior}
    if max(row['seed'] for row in prior) >= args.first_seed:
        raise ValueError('Use new training seeds above the previous export and held-out seed ranges')
    if min(len(prior_identities), len(prior_hashes)) != len(prior):
        raise ValueError('Repeated task/seed or initial state across the existing splits')
    for row in carry['episodes_detail']:
        tracks[row['task']].append(row)

    previous_paths = [row['source'] for row in carry['episodes_detail']]
    new_rows = []
    replay_failures = []

    def add_replay(screen, task, seed, destination):
        identity = task, seed
        if identity in prior_identities:
            raise ValueError(f'Repeated task and seed: {screen}')
        replay = destination / 'render_screen.json'
        failure = destination.with_suffix('.replay-failed.json')
        if not replay.is_file() and not failure.is_file():
            log = destination.with_suffix('.replay.log')
            command = ['bash', 'scripts/run_wsl.sh', 'render-screen', str(screen), '--output', str(destination)]
            if f'{task}:{seed}' in args.allow_state_drift:
                command.append('--allow-state-drift')
            with log.open('w', encoding='utf-8') as file:
                process = subprocess.run(command, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT, check=False)
            if process.returncode != 0:
                failure.write_text(json.dumps({'task': task, 'seed': seed, 'exit_code': process.returncode,
                                               'log': log.as_posix()}, indent=2) + '\n', encoding='utf-8')
        if failure.is_file():
            replay_failures.append(read_json(failure))
            print(f'{task} seed {seed} failed replay. See {failure}.', flush=True)
            return
        evidence = read_json(replay)
        if not (evidence['success'] and evidence['terminal']):
            raise ValueError(f'Replay verification failed: {replay}')
        matches = evidence.get('state_trace_matches', evidence['max_abs_physics_state_error'] <= 1e-8)
        if not matches and f'{task}:{seed}' not in args.allow_state_drift:
            raise ValueError(f'Replay state differs from its source. Review {replay} and approve that seed explicitly.')
        record = read_record(destination / 'episode.h5', 'train')
        if (record['task'], record['seed'], record['frames']) != (task, seed, evidence['steps']):
            raise ValueError(f'Replay metadata differs from capture: {replay}')
        if record['initial_state_sha256'] in prior_hashes:
            raise ValueError(f'Repeated initial state: {replay}')
        prior_hashes.add(record['initial_state_sha256'])
        prior_identities.add(identity)
        tracks[task].append(record)
        new_rows.append(record)
        print(f'{task}: {len(tracks[task])}/{args.target_per_task} replay-verified training episodes', flush=True)

    for recovered in args.recovered_screen:
        screen = recovered.parent
        result = read_json(screen / 'expert.json')
        if int(screen.with_suffix('.status').read_text()) != 0 or not result['success']:
            raise ValueError(f'Expected a successful recovered screen: {screen}')
        task, seed = result['task'], result['seed']
        for group in (evaluation['ordinary'], evaluation['fruit_language']):
            first, last = group['seeds_inclusive']
            if task in group['tasks'] and first <= seed <= last:
                raise ValueError(f'The recovered screen is a reserved evaluation case: {recovered}')
        add_replay(recovered, task, seed, args.batch / f'{task}-{seed}-recovered-rgb')

    selected_tasks = args.task or TASKS
    for seed in range(args.first_seed, args.last_seed + 1):
        if all(len(tracks[task]) >= args.target_per_task for task in selected_tasks):
            break
        for task in selected_tasks:
            if len(tracks[task]) >= args.target_per_task:
                continue
            screen = args.batch / f'{task}-{seed}-screen'
            status = screen.with_suffix('.status')
            if not status.is_file():
                run('screen_expansion_01_wsl.sh', args.batch, task, seed)
            if int(status.read_text()) != 0:
                continue
            destination = args.batch / f'{task}-{seed}-rgb'
            add_replay(screen / 'screening.h5', task, seed, destination)

    episodes = previous_paths + [row['source'] for row in new_rows]
    output = Path('data') / f'{args.batch.name}-v21'
    verification = output / 'verification.json'
    options = ['--openpi', '--action-horizon', '50']
    if verification.is_file():
        options.append('--verify-only')
    for episode in episodes:
        options.extend(['--episode', f'train={episode}'])
    if not new_rows:
        print('No new replay-verified episodes in this seed range', flush=True)
        train = carry
    elif args.scratch_root is not None and not verification.is_file():
        publishing = output.with_suffix('.publishing')
        if output.exists() or publishing.exists():
            raise FileExistsError(f'An incomplete dataset already exists under {output.parent}')
        scratch = Path(tempfile.mkdtemp(prefix=f'{args.batch.name}-', dir=args.scratch_root))
        staged = scratch / output.name
        print(f'Writing the dataset on the WSL filesystem at {staged}', flush=True)
        base = args.previous.parent
        if (base / 'train' / 'meta' / 'info.json').is_file():
            options.extend(['--base-export', base])
        run('convert_lerobot_wsl.sh', *options, '--input-scratch', scratch, '--output', staged)
        shutil.move(staged, publishing)
        publishing.rename(output)
        scratch.rmdir()
    else:
        run('convert_lerobot_wsl.sh', *options, '--output', output)
    if new_rows:
        train = read_json(verification)['splits']['train']
    if not train['all_frames_verified'] or [row['source'] for row in train['episodes_detail']] != episodes:
        raise ValueError('The export differs from the verified episode list')
    attempts = {task: Counter() for task in TASKS}
    screening_records = []
    for task in TASKS:
        for p in sorted(args.batch.glob(f'{task}-*-screen.status')):
            seed = int(p.name.removeprefix(f'{task}-').removesuffix('-screen.status'))
            if args.first_seed <= seed <= args.last_seed:
                screening = screen_result(task, seed, batch=args.batch)
                screening_records.append(screening)
                attempts[task][screening['outcome']] += 1
    report = {
        'training_split': train,
        'new_rgb_replays': [read_json(Path(row['source']).with_name('render_screen.json')) for row in new_rows],
        'training_episodes_by_task': {task: len(rows) for task, rows in tracks.items()},
        'training_frames_by_task': {task: sum(row['frames'] for row in rows) for task, rows in tracks.items()},
        'training_candidate_seed_range_inclusive': [args.first_seed, args.last_seed],
        'target_episodes_per_task': args.target_per_task,
        'selected_tasks': list(selected_tasks),
        'allowed_state_drift': args.allow_state_drift,
        'successful_retries': [p.as_posix() for p in args.recovered_screen],
        'screening_outcomes_by_task': attempts,
        'screening_attempts': screening_records,
        'replay_failures': replay_failures,
        'checked_distinct_initializations_including_held_out': len(prior_hashes),
    }
    args.batch.mkdir(parents=True, exist_ok=True)
    report_path = args.batch / 'dataset_report.json'
    partial = report_path.with_suffix('.tmp')
    partial.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    partial.replace(report_path)
    print(json.dumps({key: value for key, value in report.items() if key not in ('training_split', 'screening_attempts', 'new_rgb_replays')}, indent=2), flush=True)
    if new_rows:
        print(f'{train["episodes"]} episodes, {train["frames"]} frames verified in {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path, help='Verification report for the previous training export')
    parser.add_argument('batch', type=Path, help='Path under runs/stage2-*')
    parser.add_argument('first_seed', type=int)
    parser.add_argument('last_seed', type=int)
    parser.add_argument('--task', choices=TASKS, action='append', help='Task to screen. Defaults to all three.')
    parser.add_argument('--allow-state-drift', action='append', default=[], metavar='TASK:SEED',
                        help='Accept a successful replay with different states for this reviewed task and seed')
    parser.add_argument('--target-per-task', type=int, default=20)
    parser.add_argument('--scratch-root', type=Path, help='Stage the verified export on the native Linux filesystem')
    parser.add_argument('--recovered-screen', type=Path, action='append', default=[],
                        help='Successful retry to replay separately from fixed-seed screening')
    args = parser.parse_args()
    if not re.fullmatch(r'runs/stage2-[a-z0-9-]+', args.batch.as_posix()):
        parser.error('Batch path must have the form runs/stage2-NAME')
    if not 0 <= args.first_seed <= args.last_seed < 2**32 or args.target_per_task < 1:
        parser.error('Use a valid seed range and a positive target')
    for episode in args.allow_state_drift:
        task, _, seed = episode.partition(':')
        if task not in TASKS or not seed.isdecimal():
            parser.error(f'Invalid task and seed for --allow-state-drift: {episode}')
    if Path.cwd() != ROOT:
        parser.error(f'Run from {ROOT}')
    extend(args)


if __name__ == '__main__':
    main()
