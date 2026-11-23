"""Compare fixed-noise flow fit on physical coordinates and unused padded outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pi05_runtime import file_sha256, restore_trainable
from reference_pose_lerobot import load_pose_split
from train_reference_aligned_fruit import load_aligned_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--case', required=True, metavar='TASK:SEED')
    args = parser.parse_args()
    task, seed = args.case.rsplit(':', 1)
    episode, = [ep for ep in load_pose_split(args.dataset, 'train')
                if (ep.task, ep.seed) == (task, int(seed))]
    policy = load_aligned_policy(args.run[0], args.base, args.fork, stage='base')
    report = dict(kind='fixed_expert_observation_flow_fit_not_autonomous_success',
                  case=args.case, optimizer_updates=0, policy_rollouts_executed=0,
                  dataset_verification_sha256=file_sha256(args.dataset / 'verification.json'),
                  script_sha256=file_sha256(__file__), stages={})
    frames = np.linspace(0, len(episode.state) - 1, 12, dtype=int)
    with torch.inference_mode():
        for name, run in [('published_reference', None), *[(str(run), run) for run in args.run]]:
            if run is not None:
                checkpoint = torch.load(run / 'sft.pt', map_location='cpu', weights_only=True)
                metadata = checkpoint['metadata']
                if (metadata['norm_stats_sha256'] != policy.provenance['norm_stats_sha256']
                        or metadata.get('shortest_euler', False) != policy.shortest_euler
                        or metadata['base_sha256'] != policy.provenance['base_sha256']):
                    raise ValueError('Only compare checkpoints with matching transforms and parents')
                restore_trainable(policy.model, checkpoint['policy'])
                del checkpoint
            sums = torch.zeros((10, 32), device='cuda')
            count = 0
            for start in range(0, len(frames), 4):
                subset = frames[start:start + 4]
                observation, labels = policy.encode_training_batch([episode.row(int(frame)) for frame in subset])
                noise = torch.randn(labels.shape, device='cuda',
                                    generator=torch.Generator(device='cuda').manual_seed(4900 + start))
                for instant in (0.1, 0.5, 0.9):
                    loss = policy.model.compute_loss(observation, labels, train=False, noise=noise,
                                                    time=torch.full((len(subset),), instant, device='cuda'))
                    sums += loss.sum(0)
                    count += len(subset)
            error = (sums / count).float().cpu().numpy()
            row = dict(frames=frames.tolist(), noise_time_evaluations=count,
                       per_coordinate_mean_squared_flow_error=error.mean(0).tolist(),
                       xyz=float(error[:, :3].mean()), euler=float(error[:, 3:6].mean()),
                       gripper=float(error[:, 6].mean()), physical_seven=float(error[:, :7].mean()),
                       executed_five_physical_seven=float(error[:5, :7].mean()),
                       unused_padding_twenty_five=float(error[:, 7:].mean()), all_thirty_two=float(error.mean()))
            if run is not None:
                row['sft_sha256'] = file_sha256(run / 'sft.pt')
            report['stages'][name] = row
            print(json.dumps(dict(stage=name, **row)), flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
