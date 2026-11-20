"""Fit OpenPI statistics on the local LeRobot v2.1 training pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

from convert_lerobot import FPS, JOINTS
from vlabench_policy_io import ACTION_HORIZON, ACTUATOR_DIM


def read_training_vectors(root):
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
        windows.append(action[np.minimum(indices, count - 1)].reshape(-1, ACTUATOR_DIM))
    state, actions = np.concatenate(states), np.concatenate(windows)
    if len(state) != info['total_frames'] or len(actions) != len(state) * ACTION_HORIZON:
        raise ValueError('Training frame or action window count differs from metadata')
    return {'state': state, 'actions': actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    stats = {}
    for key, values in read_training_vectors(args.train_root).items():
        accumulator = normalize.RunningStats()
        accumulator.update(values)
        stats[key] = accumulator.get_statistics()
        if not np.all(stats[key].q99 > stats[key].q01):
            raise ValueError(f'{key}: degenerate normalization quantile in the training pilot')
        print(f'{key}: {len(values)} vectors of width {values.shape[-1]}')
    # Verify the exact output using the upstream deserializer before writing it.
    reloaded = normalize.deserialize_json(normalize.serialize_json(stats))
    for key, value in stats.items():
        for field in ('mean', 'std', 'q01', 'q99'):
            np.testing.assert_array_equal(getattr(reloaded[key], field), getattr(value, field))
    normalize.save(args.output_dir, stats)
    print(f'Wrote {args.output_dir / "norm_stats.json"}')


if __name__ == '__main__':
    main()
