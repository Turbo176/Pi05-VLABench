"""Compare deterministic 10-step predictions on aligned validation demo windows."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from evaluate_pi05 import load_stage
from pi05_runtime import DENOISE_STEPS
from vlabench_policy_io import ACTION_SPACES, SCENE_CAMERA, WRIST_CAMERA, simulator_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'stats', 'dataset', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--action-space', choices=ACTION_SPACES)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--stage', required=True, choices=('base','sft','ppo'))
    args = parser.parse_args()
    inputs = pq.read_table(args.dataset).to_pylist()
    model, policy, digest = load_stage(SimpleNamespace(**vars(args), execute_steps=10))
    errors = defaultdict(list)
    with torch.inference_mode():
        for index, row in enumerate(inputs):
            views = {}
            for name, camera in [('forward', SCENE_CAMERA), ('wrist', WRIST_CAMERA)]:
                with Image.open(io.BytesIO(row[name])) as image:
                    views[camera] = np.asarray(image.convert('RGB'))
            example = simulator_inputs(views, row['state'], row['prompt'])
            observation, _ = policy.encode(example)
            generator = torch.Generator(device='cuda').manual_seed(4000 + index)
            predicted = policy.decode(model.sample_actions(observation, num_steps=DENOISE_STEPS, rng=generator), example['state'])[:10]
            expert = np.array(row['actions'])
            difference = np.abs(predicted - expert)
            errors[row['task_id']].append({'frame': row['frame'], 'arm_mae_rad': float(difference[:,:7].mean()),
                                           'finger_mae_m': float(difference[:,7:].mean()),
                                           'finger_threshold_agreement': float(((predicted[:,7:] > 0.02) == (expert[:,7:] > 0.02)).mean())})
    summaries = {task: {metric: float(np.mean([r[metric] for r in records])) for metric in
                        ('arm_mae_rad','finger_mae_m','finger_threshold_agreement')} for task, records in errors.items()}
    report = {'stage': args.stage,'checkpoint_sha256':digest, 'stats_sha256':policy.stats_sha256,
              'predictions_per_task': {key:len(val) for key,val in errors.items()}, 'per_task_means':summaries,
              'predictions':errors}
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries),flush=True)


if __name__ == '__main__':
    main()
