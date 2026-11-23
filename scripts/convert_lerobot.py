"""Export successful VLABench episodes and validate every LeRobot frame."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

CAMERAS = ('right', 'left', 'forward', 'franka/Franka_wrist_cam')
IMAGE_KEYS = ('right', 'left', 'forward', 'wrist')
POSE_CAMERAS = ('forward', 'right', 'wrist')
POSE_TASKS = ('select_fruit', 'insert_flower', 'take_out_cool_drink')
POSE_FIELDS = ['x_m', 'y_m', 'z_m', 'roll_rad', 'pitch_rad', 'yaw_rad', 'gripper']
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


def read_pose_record(path, split):
    with h5py.File(path, 'r') as source:
        count = len(source['actions'])
        if (source.attrs['task'] not in POSE_TASKS or not count or source.attrs['fps'] != FPS
                or source['rgb'].shape != (count, 3, 224, 224, 3) or source['rgb'].dtype != np.uint8
                or not str(source.attrs['instruction']).strip()
                or json.loads(source.attrs['cameras']) != ['forward', 'right', 'franka/Franka_wrist_cam']):
            raise ValueError(f'Invalid reference-aligned episode {path}')
        for key in ('state', 'actions'):
            if source[key].shape != (count, len(POSE_FIELDS)) or not np.isfinite(source[key][:]).all():
                raise ValueError(f'Invalid pose field {key} in {path}')
        for key in ('success', 'terminal'):
            if source[key].shape != (count,) or not source[key][-1] or np.any(source[key][:-1]):
                raise ValueError(f'Expected a unique successful final step in {path}')
        times = source['time'][:]
        np.testing.assert_allclose(times - times[0], np.arange(count) / FPS, atol=1e-9, rtol=0)
        record = {'source': str(path), 'split': split, 'task': str(source.attrs['task']),
                  'seed': int(source.attrs['seed']), 'instruction': str(source.attrs['instruction']),
                  'frames': count, 'initial_state_sha256': str(source.attrs['initial_state_sha256'])}
        if 'render_samples' in source.attrs:
            record['render_samples'] = int(source.attrs['render_samples'])
        if 'policy_prefix' in source:
            prefix = source['policy_prefix'][:]
            start_hash = str(source.attrs['prefix_end_state_sha256'])
            if (split != 'train' or prefix.ndim != 2 or prefix.shape[1] != len(JOINTS)
                    or not len(prefix) or not np.isfinite(prefix).all() or len(start_hash) != 64):
                raise ValueError(f'Invalid training-only expert correction {path}')
            record.update(policy_prefix_steps=len(prefix), prefix_end_state_sha256=start_hash)
            for key in ('policy_parent_sha256', 'norm_stats_sha256', 'base_sha256'):
                if key in source.attrs:
                    record[key] = str(source.attrs[key])
        return record


def schema(pose=False):
    image_keys, image_size = (POSE_CAMERAS, 224) if pose else (IMAGE_KEYS, 256)
    coordinate_names = POSE_FIELDS if pose else JOINTS
    features = {f'observation.images.{key}': {'dtype': 'image', 'shape': (image_size, image_size, 3),
                                              'names': ['height', 'width', 'channels']}
                for key in image_keys}
    for key in ('observation.state', 'action'):
        features[key] = {'dtype': 'float32', 'shape': (len(coordinate_names),), 'names': coordinate_names}
    return features


@contextmanager
def open_source(path, scratch):
    if scratch is None:
        with h5py.File(path) as source:
            yield source
        return
    with tempfile.TemporaryDirectory(prefix='episode-', dir=scratch) as directory:
        staged = Path(directory) / 'episode.h5'
        shutil.copyfile(path, staged)
        with h5py.File(staged) as source:
            yield source


def create_split(dataset_class, root, repo_id, records, scratch=None, base_export=None, *, pose=False):
    image_keys, action_key = (POSE_CAMERAS, 'actions') if pose else (IMAGE_KEYS, 'action')
    if base_export is None:
        dataset = dataset_class.create(repo_id, fps=FPS, root=root, features=schema(pose), use_videos=False,
                                       robot_type='franka', video_backend='pyav')
    else:
        base_report = json.loads((base_export / 'verification.json').read_text(encoding='utf-8'))
        base = base_report['splits']['train']
        base_records = base['episodes_detail']
        if (base_report['format'] != 'LeRobot v2.1' or not base['all_frames_verified']
                or not 0 < len(base_records) < len(records)
                or records[:len(base_records)] != base_records):
            raise ValueError('The base export must be a verified prefix of the requested training split')
        shutil.copytree(base_export / 'train', root)
        dataset = dataset_class(repo_id, root=root, video_backend='pyav')
        if (dataset.num_episodes != len(base_records) or dataset.num_frames != base['frames']
                or dataset.fps != FPS or dataset.meta.video_keys):
            raise ValueError('The copied training split does not match the base verification report')
        dataset.hf_dataset = dataset.create_hf_dataset()
        records = records[len(base_records):]
    try:
        for record in records:
            with open_source(record['source'], scratch) as source:
                for index in range(record['frames']):
                    rgb = source['rgb'][index]
                    row = {f'observation.images.{key}': frame for key, frame in zip(image_keys, rgb)}
                    row.update({'observation.state': source['state'][index].astype(np.float32),
                                'action': source[action_key][index].astype(np.float32),
                                'task': record['instruction']})
                    dataset.add_frame(row)
            dataset.save_episode()
            if dataset.meta.info['codebase_version'] == 'v2.1':
                # The pinned writer keeps all previous Arrow tables in memory.
                # Episode offsets come from metadata, and readback reopens the files on disk.
                dataset.hf_dataset = dataset.create_hf_dataset()
    finally:
        if hasattr(dataset, 'finalize'):
            dataset.finalize()


def verify_split(dataset_class, root, repo_id, records, scratch=None, *, pose=False):
    image_keys, action_key = (POSE_CAMERAS, 'actions') if pose else (IMAGE_KEYS, 'action')
    dataset = dataset_class(repo_id, root=root, video_backend='pyav')
    expected = set(schema(pose)) | {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    if set(dataset.features) != expected or dataset.fps != FPS:
        raise ValueError(f'Dataset schema or rate mismatch at {root}: {dataset.features}')
    count = sum(record['frames'] for record in records)
    if dataset.num_episodes != len(records) or len(dataset) != count:
        raise ValueError(f'Dataset episode or frame count mismatch at {root}')
    absolute_index = 0
    for episode, record in enumerate(records):
        with open_source(record['source'], scratch) as source:
            for local_index in range(record['frames']):
                row = dataset[absolute_index]
                if (row['task'] != record['instruction'] or row['episode_index'].item() != episode
                        or row['frame_index'].item() != local_index or row['index'].item() != absolute_index):
                    raise ValueError(f'Frame indexing or instruction mismatch at {root}, row {absolute_index}')
                np.testing.assert_allclose(row['timestamp'].item(), local_index / FPS, atol=2e-6, rtol=0)
                np.testing.assert_array_equal(row['observation.state'].numpy(), source['state'][local_index].astype(np.float32))
                np.testing.assert_array_equal(row['action'].numpy(), source[action_key][local_index].astype(np.float32))
                for key, original in zip(image_keys, source['rgb'][local_index]):
                    image = row[f'observation.images.{key}'].numpy()
                    actual = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    np.testing.assert_array_equal(actual, original)
                absolute_index += 1
    return {'episodes': len(records), 'frames': count, 'all_frames_verified': True}


def verify_action_chunks(dataset_class, root, repo_id, records, horizon, *, pose=False):
    action_key = 'actions' if pose else 'action'
    offsets = np.arange(horizon)
    dataset = dataset_class(repo_id, root=root, video_backend='pyav',
                            delta_timestamps={'action': (offsets / FPS).tolist()})
    start = 0
    checks = 0
    for record in records:
        with h5py.File(record['source']) as source:
            actions = source[action_key][:].astype(np.float32)
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
    parser.add_argument('--input-scratch', type=Path, help='Copy one HDF5 at a time here for faster local reads')
    parser.add_argument('--reference-pose', action='store_true', help='Export three cameras and seven pose coordinates')
    parser.add_argument('--verify-only', action='store_true', help='Verify a previously exported dataset')
    parser.add_argument('--base-export', type=Path, help='Copy a verified v2.1 training export and append to the copy')
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
        record = (read_pose_record if args.reference_pose else read_record)(Path(path), split)
        initialization = record['initial_state_sha256']
        identity = (record['task'], record['seed'])
        if initialization in seen or identity in seeds:
            parser.error(f'Repeated initialization or task/seed: {path}')
        seen.add(initialization)
        seeds.add(identity)
        records[split].append(record)

    if args.base_export and (args.reference_pose or args.format != 'v2.1' or args.verify_only or records['validation'] or records['test']):
        parser.error('--base-export requires a new train-only LeRobot v2.1 export')
    if not args.verify_only:
        args.output.mkdir(parents=True, exist_ok=False)
    summary = {'format': f'LeRobot {args.format}', 'fps': FPS, 'splits': {}}
    if args.reference_pose:
        summary['profile'] = 'vlabench_reference_pose'
    for split, entries in records.items():
        if not entries:
            continue
        repo_id = f'local/pi05-vlabench-{split}'
        root = args.output / split
        if not args.verify_only:
            create_split(LeRobotDataset, root, repo_id, entries, args.input_scratch, args.base_export,
                         pose=args.reference_pose)
        result = verify_split(LeRobotDataset, root, repo_id, entries, args.input_scratch, pose=args.reference_pose)
        if args.action_horizon:
            result['action_chunks'] = verify_action_chunks(LeRobotDataset, root, repo_id, entries, args.action_horizon,
                                                           pose=args.reference_pose)
        summary['splits'][split] = {**result, 'episodes_detail': entries}
    (args.output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
