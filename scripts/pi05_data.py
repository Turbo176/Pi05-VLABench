"""Read the project's embedded-image LeRobot v2.1 export for single-GPU SFT."""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from vlabench_policy_io import ACTION_HORIZON, lerobot_inputs

VIEWS = ('observation.images.forward', 'observation.images.wrist')
COLUMNS = (*VIEWS, 'episode_index', 'frame_index', 'task_index', 'observation.state', 'action')
TASKS = ('select_fruit', 'insert_flower', 'take_out_cool_drink')


class LeRobotTrajectories:
    """Sample a task, then an episode and frame, without crossing episode boundaries."""

    def __init__(self, root, manifest, episode_filter=None):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text(encoding='utf-8'))
        if self.root.name != 'train' or self.info['codebase_version'] != 'v2.1' or self.info['fps'] != 10:
            raise ValueError('Expected the 10 Hz LeRobot v2.1 training split')
        self.prompts = {row['task_index']: row['task'] for row in self._jsonl('tasks')}
        self.lengths = {row['episode_index']: row['length'] for row in self._jsonl('episodes')}
        self.episodes = {task: [] for task in TASKS}
        report = json.loads(Path(manifest).read_text(encoding='utf-8'))['splits']['train']
        if not report['all_frames_verified'] or report['frames'] != self.info['total_frames']:
            raise ValueError('Training split differs from the verified export')
        for idx, entry in enumerate(report['episodes_detail']):
            task = entry['task']
            if entry['frames'] != self.lengths.get(idx):
                raise ValueError(f'Episode {idx} differs from the verified export')
            if task not in self.episodes or idx not in self.lengths:
                raise ValueError(f'Manifest has an invalid task or episode index: {entry}')
            if episode_filter is None or idx in episode_filter:
                self.episodes[task].append(idx)
        indices = [idx for group in self.episodes.values() for idx in group]
        if len(set(indices)) != len(indices) or not all(self.episodes.values()):
            raise ValueError('Expected unique episode indices and data for all three tasks')
        if episode_filter is None and set(indices) != set(self.lengths):
            raise ValueError('Manifest does not cover all training episodes')
        self._cache = {}

    def _jsonl(self, name):
        return [json.loads(line) for line in (self.root / f'meta/{name}.jsonl').read_text(encoding='utf-8').splitlines()]

    def _read(self, episode):
        if episode in self._cache:
            return self._cache[episode]
        filename = self.info['data_path'].format(episode_chunk=episode // self.info['chunks_size'], episode_index=episode)
        table = pq.ParquetFile(self.root / filename).read(columns=COLUMNS)
        length = self.lengths[episode]
        np.testing.assert_array_equal(table['episode_index'].to_numpy(), np.full(length, episode))
        np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(length))
        labels = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        if labels.shape != (length, 9) or not np.isfinite(labels).all():
            raise ValueError(f'Malformed action trajectory in {filename}')
        result = table, labels
        if len(self._cache) >= 4:
            del self._cache[next(iter(self._cache))]
        self._cache[episode] = result
        return result

    def example(self, episode, frame):
        table, labels = self._read(episode)
        if not 0 <= frame < len(labels):
            raise IndexError(frame)
        row = {k: table[k][frame].as_py() for k in (*VIEWS, 'observation.state')}
        for key in VIEWS:
            with Image.open(io.BytesIO(row[key]['bytes'])) as image:
                row[key] = np.asarray(image.convert('RGB'))
        row['task'] = self.prompts[table['task_index'][frame].as_py()]
        indices = np.minimum(frame + np.arange(ACTION_HORIZON), len(labels) - 1)
        row['action'] = labels[indices]
        return lerobot_inputs(row)

    def sample(self, rng, task):
        episode = int(rng.choice(self.episodes[task]))
        frame = int(rng.integers(self.lengths[episode]))
        return self.example(episode, frame), episode, frame
