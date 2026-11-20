"""Audit target coverage and split isolation in a verified LeRobot v2.1 export."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import h5py

from convert_lerobot import digest_arrays


def split_from_report(path, split):
    report = json.loads(path.read_text(encoding='utf-8'))['splits'][split]
    rows = report['episodes_detail']
    if (not report['all_frames_verified'] or len(rows) != report['episodes']
            or sum(row['frames'] for row in rows) != report['frames']
            or any(row['split'] != split for row in rows)):
        raise ValueError(f'Invalid verification report: {path}')
    return rows


def audit(train, validation, test):
    splits = {'train': train, 'validation': validation, 'test': test}
    identities, initializations = set(), set()
    coverage = {}
    for split, rows in splits.items():
        coverage[split] = {}
        for row in rows:
            task, seed = row['task'], row['seed']
            with h5py.File(row['source'], 'r') as episode:
                layout = digest_arrays(episode['initial_physics_state'][:],
                                       episode['initial_body_xpos'][:], episode['initial_body_xquat'][:])
                metadata = (str(episode.attrs['task']), int(episode.attrs['seed']),
                            str(episode.attrs['instruction']).strip(), len(episode['action']), layout)
                expected = (task, seed, row['instruction'], row['frames'], row['initial_state_sha256'])
                if metadata != expected:
                    raise ValueError(f'Source does not match verification: {row["source"]}')
                config = json.loads(episode.attrs['episode_config'])['task']
                target = config['target_entity']
                components = [item for item in config['components'] if item['name'] == target]
                if len(components) != 1:
                    raise ValueError(f'Expected one saved target in {row["source"]}')
                initial_position = components[0]['position']
                if len(initial_position) != 3 or not all(math.isfinite(x) for x in initial_position):
                    raise ValueError(f'Invalid saved target position in {row["source"]}')
            if (task, seed) in identities or layout in initializations:
                raise ValueError(f'Duplicate episode or initial state at {split}/{task}/{seed}')
            identities.add((task, seed))
            initializations.add(layout)
            counts = coverage[split].setdefault(task, {'episodes': 0, 'frames': 0, 'targets': Counter(),
                                                       'instructions': Counter(), 'target_xyz_m': {}})
            counts['episodes'] += 1
            counts['frames'] += row['frames']
            counts['targets'][target] += 1
            counts['instructions'][row['instruction']] += 1
            bounds = counts['target_xyz_m'].setdefault(target, {'min': initial_position, 'max': initial_position})
            bounds['min'] = [min(previous, value) for previous, value in zip(bounds['min'], initial_position)]
            bounds['max'] = [max(previous, value) for previous, value in zip(bounds['max'], initial_position)]
    return {'distinct_episode_identities': len(identities),
            'distinct_initializations': len(initializations), 'by_split_and_task': coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path, help='Expansion dataset_report.json')
    parser.add_argument('verification', type=Path, help='Expansion LeRobot verification.json')
    parser.add_argument('output', type=Path, help='Output JSON report')
    args = parser.parse_args()
    evidence = json.loads(args.batch.read_text(encoding='utf-8'))
    verified = json.loads(args.verification.read_text(encoding='utf-8'))['splits']['train']
    if verified != evidence['training_split']:
        raise ValueError('Published LeRobot verification and expansion report differ')
    chunks = verified['action_chunks']
    if chunks['horizon'] != 50 or chunks['positions_verified'] != 3 * verified['episodes']:
        raise ValueError('Incomplete action window verification')
    evidence['coverage'] = audit(
        split_from_report(args.verification, 'train'),
        split_from_report(Path('docs/stage2-three-task-validation-v21.json'), 'validation'),
        split_from_report(Path('docs/stage2-lerobot-v21.json'), 'test'),
    )
    if evidence['coverage']['distinct_initializations'] != evidence['checked_distinct_initializations_including_held_out']:
        raise ValueError('Expansion and audit initialization counts differ')
    by_task = evidence['coverage']['by_split_and_task']['train']
    for task, counts in by_task.items():
        if (counts['episodes'] != evidence['training_episodes_by_task'][task]
                or counts['frames'] != evidence['training_frames_by_task'][task]):
            raise ValueError(f'Expansion and audit counts differ for {task}')
    stats = args.verification.parent / 'train-stats' / 'norm_stats.json'
    evidence['normalization_sha256'] = hashlib.sha256(stats.read_bytes()).hexdigest()
    pending = args.output.with_suffix('.tmp')
    pending.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    pending.replace(args.output)
    print(json.dumps(evidence['coverage'], indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
