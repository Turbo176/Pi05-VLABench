"""Export successful VLABench episodes and validate every LeRobot frame."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

CAMERAS = ('right', 'left', 'forward', 'franka/Franka_wrist_cam')
IMAGE_KEYS = ('right', 'left', 'forward', 'wrist')
JOINTS = [f'joint{i}' for i in range(1, 8)] + ['finger_joint1', 'finger_joint2']
SPLITS = ('train', 'validation', 'test')
TASKS = {'select_fruit', 'insert_flower', 'add_condiment', 'get_coffee', 'take_out_cool_drink'}
FPS = 10


def digest_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        normalized = np.ascontiguousarray(array, dtype='<f8')
        digest.update(repr(normalized.shape).encode('ascii'))
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def read_record(path, split):
    with h5py.File(path) as f:
        seed = int(f.attrs['seed'])
        task = str(f.attrs['task'])
        instruction = str(f.attrs['instruction']).strip()
        count = len(f['action'])
        if task not in TASKS or f.attrs['robot'] != 'franka' or not instruction or not count:
            raise ValueError(f'Invalid episode metadata in {path}')
        if json.loads(f.attrs['camera_names']) != list(CAMERAS):
            raise ValueError(f'Unexpected camera order in {path}')
        if f['rgb'].dtype != np.uint8 or f['rgb'].shape != (count, 4, 256, 256, 3):
            raise ValueError(f'Invalid RGB frames in {path}')
        for key in ('state', 'action'):
            if f[key].shape != (count, len(JOINTS)) or not np.isfinite(f[key][:]).all():
                raise ValueError(f'Invalid {key} in {path}')
        for key in ('success', 'terminal', 'time', 'next_time'):
            if f[key].shape != (count,):
                raise ValueError(f'Invalid {key} in {path}')
        t, next_t = f['time'][:], f['next_time'][:]
        if not np.isfinite(t).all() or not np.isfinite(next_t).all():
            raise ValueError(f'Invalid time in {path}')
        if not np.isclose(f.attrs['control_dt_s'], 1/FPS, atol=1e-9, rtol=0):
            raise ValueError(f'Unsupported control frequency in {path}')
        np.testing.assert_allclose(next_t[:-1], t[1:], atol=1e-9, rtol=0)
        np.testing.assert_allclose(t - t[0], np.arange(count) / FPS, atol=1e-9, rtol=0)
        np.testing.assert_allclose(next_t - t, 1/FPS, atol=1e-9, rtol=0)
        if any(not f[k][-1] or np.any(f[k][:-1]) for k in ('success', 'terminal')):
            raise ValueError(f'Episode never reached a unique successful terminal frame: {path}')
        # Include world-space placement of static geometry as well as free joints.
        init_hash = digest_arrays(f['initial_physics_state'][:], f['initial_body_xpos'][:],
                                  f['initial_body_xquat'][:])
    return {'source': str(path), 'split': split, 'task': task, 'seed': seed,
            'instruction': instruction, 'frames': count, 'initial_state_sha256': init_hash}


def schema():
    features = {f'observation.images.{key}': {'dtype': 'image', 'shape': (256, 256, 3),
                                              'names': ['height', 'width', 'channels']}
                for key in IMAGE_KEYS}
    for key in ('observation.state', 'action'):
        features[key] = {'dtype': 'float32', 'shape': (len(JOINTS),), 'names': JOINTS}
    return features


def create_split(dataset_class, root, repo_id, records):
    dataset = dataset_class.create(repo_id, fps=FPS, root=root, features=schema(), use_videos=False,
                                    robot_type='franka', video_backend='pyav')
    try:
        for record in records:
            with h5py.File(record['source']) as source:
                for index in range(record['frames']):
                    rgb = source['rgb'][index]
                    row = {f'observation.images.{key}': frame for key, frame in zip(IMAGE_KEYS, rgb)}
                    row.update({'observation.state': source['state'][index].astype(np.float32),
                                'action': source['action'][index].astype(np.float32),
                                'task': record['instruction']})
                    dataset.add_frame(row)
            dataset.save_episode()
    finally:
        if hasattr(dataset, 'finalize'):
            dataset.finalize()


def verify_split(dataset_class, root, repo_id, records):
    dataset = dataset_class(repo_id, root=root, video_backend='pyav')
    expected = set(schema()) | {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    if set(dataset.features) != expected or dataset.fps != FPS:
        raise ValueError(f'Dataset schema or rate mismatch at {root}: {dataset.features}')
    count = sum(record['frames'] for record in records)
    if dataset.num_episodes != len(records) or len(dataset) != count:
        raise ValueError(f'Dataset episode or frame count mismatch at {root}')
    absolute_index = 0
    for episode, record in enumerate(records):
        with h5py.File(record['source']) as source:
            for local_index in range(record['frames']):
                row = dataset[absolute_index]
                if (row['task'] != record['instruction'] or row['episode_index'].item() != episode
                        or row['frame_index'].item() != local_index or row['index'].item() != absolute_index):
                    raise ValueError(f'Frame indexing or instruction mismatch at {root}, row {absolute_index}')
                np.testing.assert_allclose(row['timestamp'].item(), local_index / FPS, atol=2e-6, rtol=0)
                np.testing.assert_array_equal(row['observation.state'].numpy(), source['state'][local_index].astype(np.float32))
                np.testing.assert_array_equal(row['action'].numpy(), source['action'][local_index].astype(np.float32))
                for key, original in zip(IMAGE_KEYS, source['rgb'][local_index]):
                    image = row[f'observation.images.{key}'].numpy()
                    actual = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    np.testing.assert_array_equal(actual, original)
                absolute_index += 1
    return {'episodes': len(records), 'frames': count, 'all_frames_verified': True}


def verify_action_chunks(dataset_class, root, repo_id, records, horizon):
    offsets = np.arange(horizon)
    dataset = dataset_class(repo_id, root=root, video_backend='pyav',
                            delta_timestamps={'action': (offsets / FPS).tolist()})
    start = 0
    checks = 0
    for record in records:
        with h5py.File(record['source']) as source:
            actions = source['action'][:].astype(np.float32)
            for first in sorted({0, max(0, len(actions) - 2), len(actions) - 1}):
                next_indices = first + offsets
                padded = next_indices >= len(actions)
                row = dataset[start + first]
                np.testing.assert_array_equal(row['action'].numpy(), actions[np.minimum(next_indices, len(actions) - 1)])
                np.testing.assert_array_equal(row['action_is_pad'].numpy(), padded)
                checks += 1
        start += record['frames']
    return {'horizon': horizon, 'positions_verified': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', action='append', required=True, metavar='SPLIT=FILE')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('v3.0', 'v2.1'), default='v3.0')
    parser.add_argument('--action-horizon', type=int, help='Check a window of future actions at each episode boundary')
    parser.add_argument('--verify-only', action='store_true', help='Verify a previously exported dataset')
    args = parser.parse_args()
    if args.action_horizon is not None and args.action_horizon < 1:
        parser.error('--action-horizon must be positive')
    if args.format == 'v2.1':
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    else:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    if CODEBASE_VERSION != args.format:
        parser.error(f'Installed LeRobot dataset version is {CODEBASE_VERSION}, expected {args.format}')
    records = {split: [] for split in SPLITS}
    seen = set()
    seeds = set()
    for item in args.episode:
        split, sep, path = item.partition('=')
        if split not in SPLITS or not sep or not path:
            parser.error(f'Expected SPLIT=FILE for one of {SPLITS}: {item}')
        record = read_record(Path(path), split)
        initialization = record['initial_state_sha256']
        identity = (record['task'], record['seed'])
        if initialization in seen or identity in seeds:
            parser.error(f'Repeated initialization or task/seed: {path}')
        seen.add(initialization)
        seeds.add(identity)
        records[split].append(record)

    if not args.verify_only:
        args.output.mkdir(parents=True, exist_ok=False)
    summary = {'format': f'LeRobot {args.format}', 'fps': FPS, 'splits': {}}
    for split, entries in records.items():
        if not entries:
            continue
        repo_id = f'local/pi05-vlabench-{split}'
        root = args.output / split
        if not args.verify_only:
            create_split(LeRobotDataset, root, repo_id, entries)
        result = verify_split(LeRobotDataset, root, repo_id, entries)
        if args.action_horizon:
            result['action_chunks'] = verify_action_chunks(LeRobotDataset, root, repo_id, entries, args.action_horizon)
        summary['splits'][split] = {**result, 'episodes_detail': entries}
    (args.output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
