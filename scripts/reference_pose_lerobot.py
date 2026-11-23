"""Read the verified three-camera LeRobot v2.1 pose export."""
from functools import lru_cache
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from convert_lerobot import POSE_CAMERAS, POSE_FIELDS, POSE_TASKS

IMAGE_KEYS = ('observation/image', 'observation/second_image', 'observation/wrist_image')
HORIZON = 10
IMAGE_CACHE_EPISODES = int(os.environ.get('PI05_POSE_IMAGE_CACHE', '4'))
if IMAGE_CACHE_EPISODES < 1:
    raise ValueError('PI05_POSE_IMAGE_CACHE must be a positive number of episodes')


@lru_cache(maxsize=IMAGE_CACHE_EPISODES)
def _images(path):
    return pq.ParquetFile(path).read(columns=[f'observation.images.{name}' for name in POSE_CAMERAS])


class PoseEpisode:
    def __init__(self, root, info, episode_index, record, prompt_by_task):
        self.task = record['task']
        if self.task not in POSE_TASKS:
            raise ValueError(f'Unsupported pose task {self.task}')
        self.seed = int(record['seed'])
        self.prompt = record['instruction']
        self.initial_state = record['initial_state_sha256']
        self.policy_prefix_steps = int(record.get('policy_prefix_steps', 0))
        self.render_samples = record.get('render_samples')
        self.path = root / info['data_path'].format(episode_chunk=episode_index // info['chunks_size'],
                                                   episode_index=episode_index)
        table = pq.ParquetFile(self.path).read(columns=['observation.state', 'action', 'episode_index',
                                                       'frame_index', 'timestamp', 'task_index'])
        length = record['frames']
        np.testing.assert_array_equal(table['episode_index'].to_numpy(), np.full(length, episode_index))
        np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(length))
        np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(length) / 10, rtol=0, atol=2e-6)
        if {prompt_by_task[i] for i in table['task_index'].to_pylist()} != {self.prompt}:
            raise ValueError(f'Task instruction differs from source {self.path}')
        self.state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        self.action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        if any(array.shape != (length, len(POSE_FIELDS)) or not np.isfinite(array).all()
               for array in (self.state, self.action)):
            raise ValueError(f'Malformed pose arrays in {self.path}')
        indices = np.minimum(np.arange(length)[:, None] + np.arange(HORIZON), length - 1)
        self.chunks = self.action[indices]

    def row(self, frame):
        images = _images(self.path)
        observation = {'observation/state': self.state[frame], 'prompt': self.prompt, 'actions': self.chunks[frame]}
        for camera, key in zip(POSE_CAMERAS, IMAGE_KEYS, strict=True):
            entry = images[f'observation.images.{camera}'][frame].as_py()
            with Image.open(io.BytesIO(entry['bytes'])) as image:
                observation[key] = np.array(image.convert('RGB'))
        return observation


def load_pose_split(root, split):
    root = Path(root)
    manifest = json.loads((root / 'verification.json').read_text(encoding='utf-8'))
    report = manifest['splits'][split]
    root = root / split
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    if (manifest['profile'] != 'vlabench_reference_pose' or info['fps'] != 10
            or info['codebase_version'] != 'v2.1' or not report['all_frames_verified']
            or report['action_chunks']['horizon'] != HORIZON or info['total_frames'] != report['frames']):
        raise ValueError(f'Expected a verified pose export at {root}')
    prompts = {row['task_index']: row['task'] for row in map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
    results = [PoseEpisode(root, info, index, row, prompts) for index, row in enumerate(report['episodes_detail'])]
    if len(results) != info['total_episodes'] or len({(item.task, item.seed) for item in results}) != len(results):
        raise ValueError(f'Unexpected episode count or duplicate seed at {root}')
    return results
