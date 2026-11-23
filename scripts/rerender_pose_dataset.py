"""Execute saved expert pose actions again and export aligned camera frames."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np
from openpi_client.image_tools import resize_with_pad
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

from convert_lerobot import read_pose_record
from export_reference_fruit_views import CAMERAS
from select_fruit import camera_names, digest_arrays, open_episode, physics_state, success


def replay(record, output, render_samples):
    source_path = Path(record['source'])
    with h5py.File(source_path, 'r') as source:
        if 'policy_prefix' in source:
            raise ValueError('This export requires complete expert demonstrations without a policy prefix')
        actions = source['actions'][:]
        reference_states = source['state'][:]
    name = f"{record['split']}-{record['task']}-{record['seed']}.h5"
    destination = output / name
    partial = destination.with_suffix('.partial.h5')
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    with open_episode(record['seed'], record['task'], render_samples=render_samples) as env, h5py.File(partial, 'x') as target:
        initial = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
        if (initial != record['initial_state_sha256'] or env.task.get_instruction() != record['instruction']
                or success(env) or abs(env.control_timestep() - 0.1) > 1e-8):
            raise ValueError(f'The source reset or instruction changed for {source_path}')
        indices = [camera_names(env).index(name) for name in CAMERAS]
        images = target.create_dataset('rgb', (len(actions), 3, 224, 224, 3), 'u1',
                                       maxshape=(None, 3, 224, 224, 3),
                                       chunks=(1, 3, 224, 224, 3), compression='lzf')
        origin = np.asarray(env.get_robot_frame_position())
        states, times = [], []
        controller_hash = hashlib.sha256()
        max_state_difference = 0.0
        ik_failures = 0
        for index, label in enumerate(actions):
            measured = env.robot.get_ee_state(env.physics)
            state = np.r_[measured[:3] - origin, quaternion_to_euler(measured[3:7]), measured[-1]]
            states.append(state.astype(np.float32))
            times.append(float(env.physics.data.time))
            max_state_difference = max(max_state_difference, float(np.max(np.abs(state - reference_states[index]))))
            images[index] = [resize_with_pad(env.render(camera_id=i, **env.render_options), 224, 224) for i in indices]
            ok, arm = env.robot.get_qpos_from_ee_pos(env.physics, label[:3] + origin,
                                                     euler_to_quaternion(*label[3:6]))
            ik_failures += int(not ok)
            command = np.r_[np.asarray(arm).reshape(-1), np.repeat(0.04 if label[6] >= 0.1 else 0., 2)]
            controller_hash.update(command.tobytes())
            timestep = env.step(command)
            if success(env) or timestep.last():
                break
        solved = success(env)
        count = len(states)
        if count != len(actions):
            images.resize(count, axis=0)
        # Each image and measured state precedes the action at the same index.
        for key, array in (('state', states), ('actions', actions[:count]), ('time', times)):
            target.create_dataset(key, data=np.asarray(array))
        final = np.zeros(count, dtype=bool)
        final[-1] = solved
        target.create_dataset('success', data=final)
        target.create_dataset('terminal', data=final)
        target.attrs.update(seed=record['seed'], task=record['task'], instruction=record['instruction'],
                            initial_state_sha256=initial, cameras=json.dumps(CAMERAS), fps=10,
                            render_samples=render_samples, source=str(source_path),
                            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                            action_sha256=controller_hash.hexdigest())
    if solved:
        partial.rename(destination)
    return dict(source=str(source_path), output=str(destination) if solved else str(partial),
                split=record['split'], task=record['task'], seed=record['seed'], success=solved,
                frames=count, source_frames=len(actions), initial_state_sha256=initial,
                max_source_state_difference=max_state_difference, ik_nonconverged_steps=ik_failures,
                render_samples=render_samples, action_sha256=controller_hash.hexdigest(),
                wall_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-samples', type=int, choices=(0, 2, 4, 8), required=True)
    parser.add_argument('--limit-per-task', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--source-overrides', type=Path,
                        help='JSON mapping TASK/SEED to repaired expert files with the same reset and instruction')
    args = parser.parse_args()
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error('The per-task limit must be positive')
    manifest = json.loads((args.dataset / 'verification.json').read_text(encoding='utf-8'))
    if manifest.get('profile') != 'vlabench_reference_pose':
        parser.error('Expected a verified pose export')
    overrides = json.loads(args.source_overrides.read_text(encoding='utf-8')) if args.source_overrides else {}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    log = args.output / 'episodes.jsonl'
    completed = {}
    if args.resume and log.exists():
        for row in map(json.loads, log.read_text(encoding='utf-8').splitlines()):
            key = (row['split'], row['task'], row['seed'])
            if key in completed or row['render_samples'] != args.render_samples:
                raise ValueError('Incompatible or duplicate previous rerender record')
            if not row['success']:
                raise ValueError('Review the failed replay before resuming this export')
            completed[key] = row
    rows, counts, pending = [], Counter(), []
    with log.open('a' if args.resume else 'x', encoding='utf-8') as stream:
        for split, report in manifest['splits'].items():
            if not report['all_frames_verified']:
                raise ValueError('The source dataset did not pass frame verification')
            for record in report['episodes_detail']:
                replacement = overrides.get(f"{record['task']}/{record['seed']}")
                if replacement is not None:
                    repaired = read_pose_record(Path(replacement), split)
                    for field in ('task', 'seed', 'instruction', 'initial_state_sha256'):
                        if record[field] != repaired[field]:
                            raise ValueError(f'A repaired episode changed its {field}')
                    record = repaired
                group = (split, record['task'])
                if args.limit_per_task is not None and counts[group] >= args.limit_per_task:
                    continue
                counts[group] += 1
                key = (split, record['task'], record['seed'])
                row = completed.get(key)
                if row is not None and (row['source'] != record['source'] or not Path(row['output']).is_file()):
                    raise ValueError('A resumed replay no longer matches its source or output')
                if row is not None:
                    rows.append(row)
                else:
                    pending.append(record)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            # Submit one bounded group at a time so a failed replay stops admission promptly.
            for start in range(0, len(pending), args.workers):
                batch = pending[start:start + args.workers]
                jobs = [pool.submit(replay, record, args.output, args.render_samples) for record in batch]
                failures = []
                for job in jobs:
                    row = job.result()
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    print(json.dumps(row), flush=True)
                    rows.append(row)
                    if not row['success']:
                        failures.append(row['source'])
                if failures:
                    raise RuntimeError(f'Expert action replay failed for {failures}; retained the partial episodes')
    summary = dict(source_dataset=str(args.dataset), render_samples=args.render_samples,
                   episodes=len(rows), frames=sum(row['frames'] for row in rows), all_successful=True,
                   groups={f'{split}/{task}': count for (split, task), count in counts.items()})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
