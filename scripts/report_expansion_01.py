"""Summarize every screen and verify isolation of the expanded LeRobot split."""
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


BATCH = Path('runs/stage2-expansion-01')
TRAIN = Path('data/stage2-expansion-01-v21/verification.json')
TASKS = ('select_fruit', 'insert_flower', 'take_out_cool_drink')
SEEDS = (53, 54, 55)
BUDGET = 600
TIMEOUT_SECONDS = 240


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def screen_result(task, seed, batch=BATCH):
    run = batch / f'{task}-{seed}-screen'
    exit_code = int(run.with_suffix('.status').read_text())
    log = run.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
    summary = run / 'expert.json'
    result = {'task': task, 'seed': seed, 'exit_code': exit_code}
    if summary.is_file():
        raw = read(summary)
        assert raw['task'] == task and raw['seed'] == seed
        assert raw['success'] == (exit_code == 0)
        outcome = raw.get('outcome', 'success' if raw['success'] else 'expert_unsuccessful')
        result.update(outcome=outcome, steps=raw['steps'], instruction=raw['instruction'],
                      stages=raw['stages'], wall_seconds=raw['wall_seconds'], record=summary.as_posix())
    else:
        result['record'] = run.with_suffix('.log').as_posix()
        if exit_code == 124:
            result.update(outcome='timeout', timeout_seconds=TIMEOUT_SECONDS)
        elif exit_code == 137:
            # Both an OOM kill and timeout's SIGKILL escalation can return 137.
            result.update(outcome='process_killed', signal='SIGKILL')
        elif exit_code == 139:
            result.update(outcome='simulator_process_crash', signal='SIGSEGV')
        elif 'PhysicsError:' in log:
            result.update(outcome='physics_error', diagnostic=log.splitlines()[-1])
        elif 'attempted to exceed the action budget' in log:
            with h5py.File(run / 'episode.partial.h5') as source:
                assert source['action'].shape[0] == BUDGET and not np.any(source['success'][:])
                result.update(outcome='action_budget_exhausted', steps=BUDGET,
                              instruction=str(source.attrs['instruction']))
        else:
            messages = [line for line in log.splitlines() if line.strip()]
            result.update(outcome='process_error', diagnostic=messages[-1] if messages else 'Empty log')
        skills = [line.removeprefix('Running official skill ') for line in log.splitlines()
                  if line.startswith('Running official skill ')]
        if skills:
            result['last_skill_started'] = skills[-1]
    return result


def main():
    screens = [screen_result(task, seed) for seed in SEEDS for task in TASKS]
    train = read(TRAIN)['splits']['train']
    assert train['all_frames_verified']
    train_keys = {(row['task'], row['seed']) for row in train['episodes_detail']}
    previous_train = read(Path('docs/stage2-three-task-train-v21.json'))['splits']['train']['episodes_detail']
    expected = {(row['task'], row['seed']) for row in previous_train} | {('take_out_cool_drink', 52)}
    expected |= {(row['task'], row['seed']) for row in screens if row['outcome'] == 'success'}
    assert train_keys == expected and len(train_keys) == len(train['episodes_detail'])
    validation = read(Path('docs/stage2-three-task-validation-v21.json'))['splits']['validation']['episodes_detail']
    test = read(Path('docs/stage2-lerobot-v21.json'))['splits']['test']['episodes_detail']
    records = train['episodes_detail'] + validation + test
    assert len(records) == len({(row['task'], row['seed']) for row in records})
    assert len(records) == len({row['initial_state_sha256'] for row in records})
    replays = []
    for row in train['episodes_detail'][len(previous_train):]:
        report = read(Path(row['source']).with_name('render_screen.json'))
        assert (row['task'], row['seed'], row['frames']) == (report['task'], report['seed'], report['steps'])
        assert report['success'] and report['terminal']
        replays.append(report)
    counts = Counter(row['task'] for row in train['episodes_detail'])
    frames = Counter()
    instructions = {task: Counter() for task in TASKS}
    for row in train['episodes_detail']:
        frames[row['task']] += row['frames']
        instructions[row['task']][row['instruction']] += 1
    result = {
        'action_budget': BUDGET, 'wall_time_limit_s': TIMEOUT_SECONDS, 'fixed_training_candidate_seeds': SEEDS,
        'screening_attempts': screens, 'new_rgb_replays': replays, 'training_split': train,
        'training_episodes_by_task': dict(counts),
        'training_frames_by_task': dict(frames),
        'training_instruction_counts': instructions,
        'unique_task_seed_and_initial_state_across_pilot_splits': len(records),
    }
    path = Path('docs/stage2-expansion-01.json')
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for task in TASKS:
        task_screens = [row for row in screens if row['task'] == task]
        print(task, dict(Counter(row['outcome'] for row in task_screens)), 'train', counts[task], 'episodes,', frames[task], 'frames')
    print(f'Train: {train["episodes"]} episodes, {train["frames"]} frames. Checked {len(records)} split identities.')


if __name__ == '__main__':
    main()
