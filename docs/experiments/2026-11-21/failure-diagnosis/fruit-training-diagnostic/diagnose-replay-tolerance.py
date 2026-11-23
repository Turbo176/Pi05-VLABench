import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from select_fruit import open_episode, success

manifest = Path('incoming/train-verification.json')
episodes = json.loads(manifest.read_text())['splits']['train']['episodes_detail']
training = Path('.local/compact-data/train')
info = json.loads((training / 'meta/info.json').read_text())
for seed in (42, 53):
    index, episode = next((i, e) for i, e in enumerate(episodes) if e['task'] == 'select_fruit' and e['seed'] == seed)
    path = training / info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
    labels = np.asarray(pq.ParquetFile(path).read(columns=['action'])['action'].to_pylist())
    for std in (0.0, 0.01, 0.03, 0.06, 0.10):
        for noise_seed in range(3 if std else 1):
            rng = np.random.default_rng(noise_seed)
            offsets = rng.normal(0, std, (len(labels) + 9, 7))[::10].repeat(10, axis=0)[:len(labels)]
            controls = labels.copy()
            controls[:, :7] += offsets
            with open_episode(seed, 'select_fruit') as env:
                if env.task.get_instruction() != episode['instruction'] or success(env):
                    raise ValueError('Unexpected replay reset')
                for step, control in enumerate(controls, 1):
                    if env.step(control).last():
                        break
                result = success(env)
            print(json.dumps(dict(seed=seed, noise_seed=noise_seed, bias_std_rad=std,
                                  mean_abs_bias_rad=float(abs(offsets).mean()), official_success=result,
                                  steps=step, max_steps=len(labels))), flush=True)
            if not std and not result:
                raise RuntimeError('The unperturbed expert action replay failed')
