"""Fit OpenPI statistics on the local LeRobot v2.1 training pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

from convert_lerobot import FPS, JOINTS
from vlabench_policy_io import ACTION_HORIZON, ACTION_SPACES, ACTUATOR_DIM, ARM_JOINTS, relative_arm_targets


def read_training_vectors(root, action_space='absolute', *, per_step=False):
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    if root.name != 'train' or info['codebase_version'] != 'v2.1' or info['fps'] != FPS:
        raise ValueError('Expected the train directory of a 10 Hz LeRobot v2.1 export')
    for key in ('observation.state', 'action'):
        if info['features'][key]['names'] != JOINTS or info['features'][key]['dtype'] != 'float32':
            raise ValueError(f'Invalid Franka feature schema for {key}')
    states, windows = [], []
    for episode in range(info['total_episodes']):
        path = root / info['data_path'].format(episode_chunk=episode // info['chunks_size'],
                                               episode_index=episode)
        rows = pq.ParquetFile(path).read(columns=['episode_index', 'frame_index', 'observation.state', 'action'])
        count = len(rows)
        np.testing.assert_array_equal(rows['episode_index'].to_numpy(), np.full(count, episode))
        np.testing.assert_array_equal(rows['frame_index'].to_numpy(), np.arange(count))
        state = np.asarray(rows['observation.state'].to_pylist(), dtype=np.float64)
        action = np.asarray(rows['action'].to_pylist(), dtype=np.float64)
        if any(x.shape != (count, ACTUATOR_DIM) or not np.isfinite(x).all() for x in (state, action)):
            raise ValueError(f'Invalid state or action in {path}')
        indices = np.arange(count)[:, None] + np.arange(ACTION_HORIZON)
        states.append(state)
        labels = action[np.minimum(indices, count - 1)]
        if action_space == 'arm_delta':
            labels = relative_arm_targets(labels, state)
        windows.append(labels)
    state, actions = np.concatenate(states), np.concatenate(windows)
    if len(state) != info['total_frames'] or actions.shape != (len(state), ACTION_HORIZON, ACTUATOR_DIM):
        raise ValueError('Training frame or action window count differs from metadata')
    return {'state': state, 'actions': actions if per_step else actions.reshape(-1, ACTUATOR_DIM)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--action-space', choices=ACTION_SPACES, default='absolute')
    parser.add_argument('--per-step-action-stats', action='store_true',
                        help='Fit separate arm-delta quantiles at each position in the action chunk')
    args = parser.parse_args()
    if args.per_step_action_stats and args.action_space != 'arm_delta':
        parser.error('--per-step-action-stats requires --action-space arm_delta')
    stats = {}
    vectors = read_training_vectors(args.train_root, args.action_space, per_step=args.per_step_action_stats)
    for key, values in vectors.items():
        if key == 'actions' and args.per_step_action_stats:
            # Keep finger calibration shared across the chunk. Finger targets are absolute.
            arm = values[..., :ARM_JOINTS]
            fingers = values[..., ARM_JOINTS:].reshape(-1, ACTUATOR_DIM - ARM_JOINTS)
            arms = (arm.mean(axis=0), arm.std(axis=0), *np.quantile(arm, (0.01, 0.99), axis=0))
            hand = (fingers.mean(axis=0), fingers.std(axis=0), *np.quantile(fingers, (0.01, 0.99), axis=0))
            stats[key] = normalize.NormStats(*(np.concatenate((a, np.broadcast_to(g, (ACTION_HORIZON, g.size))), axis=-1)
                                               for a, g in zip(arms, hand)))
        else:
            accumulator = normalize.RunningStats()
            accumulator.update(values)
            stats[key] = accumulator.get_statistics()
        if not np.all(stats[key].q99 > stats[key].q01):
            raise ValueError(f'{key}: degenerate normalization quantile in the training pilot')
        print(f'{key}: fit from shape {values.shape}, stats shape {stats[key].mean.shape}')
    # Verify the exact output using the upstream deserializer before writing it.
    reloaded = normalize.deserialize_json(normalize.serialize_json(stats))
    for key, value in stats.items():
        for field in ('mean', 'std', 'q01', 'q99'):
            np.testing.assert_array_equal(getattr(reloaded[key], field), getattr(value, field))
    normalize.save(args.output_dir, stats)
    print(f'Wrote {args.output_dir / "norm_stats.json"}')


if __name__ == '__main__':
    main()
