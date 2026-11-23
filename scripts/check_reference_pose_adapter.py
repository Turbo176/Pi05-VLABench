"""Replay verified LeRobot controls through the published VLABench policy adapter."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from openpi_client.image_tools import resize_with_pad

from evaluate_vlabench_reference import build_episode, load_upstream_adapter
from reference_pose_lerobot import IMAGE_KEYS, load_pose_split
from select_fruit import open_episode


class ReplayClient:
    def __init__(self, episode, execute):
        self.episode = episode
        self.execute = execute
        self.cursor = 0
        self.state_error = 0.0
        self.state_coordinate_error = np.zeros(7)
        self.image_error = {key: 0.0 for key in IMAGE_KEYS}

    def infer(self, inputs):
        row = self.episode.row(self.cursor)
        if inputs.keys() != row.keys() - {'actions'} or inputs['prompt'] != row['prompt']:
            raise RuntimeError('The adapter and LeRobot reader disagree on policy fields or instruction')
        state_difference = np.abs(row['observation/state'] - inputs['observation/state'])
        self.state_coordinate_error = np.maximum(self.state_coordinate_error, state_difference)
        state_error = np.max(state_difference)
        self.state_error = max(self.state_error, float(state_error))
        for key in IMAGE_KEYS:
            rendered = resize_with_pad(inputs[key], 224, 224)
            difference = np.abs(row[key].astype(np.int16) - rendered.astype(np.int16))
            self.image_error[key] = max(self.image_error[key], float(difference.mean()))
        self.cursor += self.execute
        return {'actions': row['actions']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--limit-per-task', type=int)
    parser.add_argument('--state-tolerance', type=float, default=1e-4)
    parser.add_argument('--rgb-mae-tolerance', type=float, default=1.0,
                        help='Allow small physics differences from float32 in-place targets in the upstream adapter')
    parser.add_argument('--report-drift', action='store_true',
                        help='Require task success while recording, rather than rejecting, numeric replay drift')
    args = parser.parse_args()
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error('The per-task limit must be positive')
    adapter = load_upstream_adapter(args.fork)
    completed = Counter()
    for episode in load_pose_split(args.dataset, args.split):
        if args.limit_per_task is not None and completed[episode.task] >= args.limit_per_task:
            continue
        client = ReplayClient(episode, execute=5)
        with open_episode(episode.seed, episode.task, render_samples=episode.render_samples) as env:
            result, rollout = build_episode(env, adapter(client, replan_steps=client.execute), episode.seed,
                                            len(episode.state), False, task=episode.task)
            if result['initial_state_sha256'] != episode.initial_state:
                raise RuntimeError('The adapter replay did not start from the recorded reset')
            for _ in rollout:
                pass
        result.update(max_state_abs_error=client.state_error, max_rgb_frame_mae=client.image_error,
                      max_xyz_error_m=float(client.state_coordinate_error[:3].max()),
                      max_euler_error_rad=float(client.state_coordinate_error[3:6].max()),
                      max_gripper_state_error=float(client.state_coordinate_error[6]),
                      recorded_controls=len(episode.state), render_samples=episode.render_samples,
                      state_within_tolerance=client.state_error <= args.state_tolerance,
                      rgb_within_tolerance=max(client.image_error.values()) <= args.rgb_mae_tolerance)
        print(json.dumps(result), flush=True)
        if not result['success'] or (not args.report_drift and (
                not result['state_within_tolerance'] or not result['rgb_within_tolerance'])):
            raise RuntimeError('The closed-loop adapter replay differs from the verified training observations')
        completed[episode.task] += 1


if __name__ == '__main__':
    main()
