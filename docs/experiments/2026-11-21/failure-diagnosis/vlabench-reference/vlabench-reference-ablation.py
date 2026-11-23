import json
from pathlib import Path
import sys
import time

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training.config import get_config
from evaluate_vlabench_reference import build_episode, load_upstream_adapter
from select_fruit import open_episode

fork, checkpoint, paired = (Path(p) for p in sys.argv[1:])
seed = 1201
initial = next(row for row in map(json.loads, (paired / 'episodes.jsonl').read_text().splitlines()) if row['seed'] == seed)
if not initial['success']:
    raise RuntimeError('Use a reference success case as the matched control')
client = policy_config.create_trained_policy(get_config('pi05_ft_vlabench_primitive'), checkpoint)
upstream_transform = client._input_transform
adapter = load_upstream_adapter(fork)


def disable_right_camera(data):
    inputs = upstream_transform(data)
    inputs['image']['left_wrist_0_rgb'] = np.zeros_like(inputs['image']['left_wrist_0_rgb'])
    inputs['image_mask']['left_wrist_0_rgb'] = np.False_
    return inputs


with (paired / 'adapter-ablation.jsonl').open('x') as report:
    for condition, steps, transform in [('execute_10', 10, upstream_transform), ('remove_right_camera', 5, disable_right_camera)]:
        client._input_transform = transform
        client._rng = jax.random.key(seed)
        started = time.monotonic()
        with open_episode(seed, 'select_fruit') as env:
            result, rollout = build_episode(env, adapter(client, replan_steps=steps), seed, 600, False)
            for key in ('instruction', 'initial_state_sha256', 'initial_rgb_sha256'):
                if result[key] != initial[key]:
                    raise RuntimeError(f'Mismatched reference reset: {key}')
            for _ in rollout:
                pass
        result.update(condition=condition, replan_steps=steps, control_hz=10, max_control_steps=600,
                      model_revision=initial['model_revision'], wall_seconds=time.monotonic() - started)
        report.write(json.dumps(result) + '\n')
        report.flush()
        print(condition, result['success'], result['steps'], flush=True)
