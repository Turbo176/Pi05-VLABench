"""Audit real pose labels through training transforms and the deployment controller."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from openpi import transforms
from openpi.shared import normalize
from openpi_client.image_tools import resize_with_pad

from evaluate_vlabench_reference import build_episode, load_upstream_adapter
from reference_pose_lerobot import IMAGE_KEYS, load_pose_split
from select_fruit import open_episode
from train_reference_aligned_fruit import AlignedPolicy, wrap_euler


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TransformedLabelReplay:
    """Supply stored expert labels, without a model or target information."""

    def __init__(self, episode, policy):
        self.episode, self.policy = episode, policy
        self.cursor = 0
        self.maximum = Counter()

    def infer(self, observation):
        stored = self.episode.row(self.cursor)
        if observation.keys() != stored.keys() - {'actions'} or observation['prompt'] != self.episode.prompt:
            raise ValueError('Deployment and dataset policy fields differ')
        for key in IMAGE_KEYS:
            difference = np.abs(resize_with_pad(observation[key], 224, 224).astype(np.int16)
                                - stored[key].astype(np.int16))
            self.maximum[f'{key}_pixel_mae'] = max(self.maximum[f'{key}_pixel_mae'], float(difference.mean()))
        actual = {**observation, 'actions': stored['actions'].copy()}
        encoded = self.policy.inputs(actual)
        decoded = self.policy.outputs({'state': encoded['state'].copy(),
                                       'actions': encoded['actions'].astype(np.float32)})['actions']
        difference = decoded - stored['actions']
        if self.policy.shortest_euler:
            difference[..., 3:6] = wrap_euler(difference[..., 3:6])
        difference = np.abs(difference)
        self.maximum['round_trip_abs_error'] = max(self.maximum['round_trip_abs_error'], float(difference.max()))
        # Compare the actual model-ready arrays, not just the resized source images.
        training = self.policy.inputs(stored)
        for key in training['image']:
            error = np.abs(encoded['image'][key].astype(np.float32) - training['image'][key].astype(np.float32))
            self.maximum[f'{key}_model_image_mae'] = max(self.maximum[f'{key}_model_image_mae'], float(error.mean()))
        self.maximum['measured_state_error'] = max(self.maximum['measured_state_error'],
            float(np.abs(observation['observation/state'] - stored['observation/state']).max()))
        if not np.array_equal(training['tokenized_prompt'], encoded['tokenized_prompt']):
            raise ValueError('Training and deployment language tokens differ')
        if difference.max() > 2e-6:
            raise ValueError('Training transforms do not recover deployment action labels')
        self.cursor += 5
        return {'actions': decoded}


def label_statistics(episodes, policy, stats):
    summaries = {}
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    for task in sorted({episode.task for episode in episodes}):
        group = [episode for episode in episodes if episode.task == task]
        transformed, closed, transitions = [], 0, 0
        round_trip_error = 0.0
        for episode in group:
            raw = {**dict.fromkeys(IMAGE_KEYS, dummy), 'prompt': episode.prompt,
                   'observation/state': episode.state.copy(), 'actions': episode.chunks.copy()}
            prepared = policy.prepare_labels(raw)
            encoded = transforms.Normalize(stats, use_quantiles=True)(prepared)
            decoded = np.stack([policy.outputs({'state': encoded['state'][frame].copy(),
                'actions': encoded['actions'][frame].astype(np.float32)})['actions']
                for frame in range(len(episode.state))])
            difference = decoded - episode.chunks
            if policy.shortest_euler:
                difference[..., 3:6] = wrap_euler(difference[..., 3:6])
            round_trip_error = max(round_trip_error, float(np.abs(difference).max()))
            transformed.append(encoded['actions'][..., :7].reshape(-1, 7))
            closed += int((episode.action[:, 6] < 0.1).sum())
            transitions += int(np.count_nonzero(np.diff(episode.action[:, 6] >= 0.1)))
        labels = np.concatenate(transformed)
        frames = sum(len(episode.state) for episode in group)
        summaries[task] = dict(episodes=len(group), frames=frames,
            instructions=dict(Counter(episode.prompt for episode in group)),
            closed_frames=closed, closed_fraction=closed / frames, gripper_transitions=transitions,
            normalized_label_quantiles=np.quantile(labels, [0, .01, .5, .99, 1], axis=0).tolist(),
            normalized_outside_minus1_plus1_fraction=(np.abs(labels) > 1.00001).mean(axis=0).tolist(),
            all_windows_round_trip_max_abs_error=round_trip_error)
        if round_trip_error > 2e-6:
            raise ValueError(f'Label normalization round trip failed for {task}')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, action='append')
    parser.add_argument('--norm-stats', type=Path, required=True)
    parser.add_argument('--fork', type=Path, required=True)
    parser.add_argument('--case', required=True, action='append', metavar='TASK:SEED')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shortest-euler', action='store_true',
                        help='Check equivalent Euler differences modulo 2*pi through the same controller')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path')
    episodes = [episode for root in args.dataset for episode in load_pose_split(root, 'train')
                if episode.task in {'select_fruit', 'insert_flower'}]
    stats = normalize.deserialize_json(args.norm_stats.read_text(encoding='utf-8'))
    policy = AlignedPolicy(None, stats, args.fork, shortest_euler=args.shortest_euler)
    report = dict(kind='real_dataset_transform_and_controller_oracle_audit',
        model_loaded=False, optimizer_updates=0, source_sha256=checksum(__file__),
        shortest_euler=args.shortest_euler,
        norm_stats_sha256=checksum(args.norm_stats),
        dataset_verification_sha256={str(root): checksum(root / 'verification.json') for root in args.dataset},
        label_statistics=label_statistics(episodes, policy, stats), oracle_replays=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'label_statistics': report['label_statistics']}), flush=True)
    adapter = load_upstream_adapter(args.fork)
    for case in args.case:
        task, seed = case.rsplit(':', 1)
        selected = [ep for ep in episodes if (ep.task, ep.seed) == (task, int(seed))]
        if len(selected) != 1 or selected[0].policy_prefix_steps:
            raise ValueError('Select one full demonstration without a policy prefix')
        episode, = selected
        client = TransformedLabelReplay(episode, policy)
        with open_episode(episode.seed, task, render_samples=episode.render_samples) as env:
            result, rollout = build_episode(env, adapter(client, replan_steps=5), episode.seed,
                                            len(episode.state), False, task=task)
            if result['initial_state_sha256'] != episode.initial_state:
                raise ValueError('Oracle reset differs from the recorded demonstration')
            for _ in rollout:
                pass
        result.update(transform_and_input_maxima=dict(client.maximum), expert_label_oracle=True,
                      model_loaded=False)
        report['oracle_replays'].append(result)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result), flush=True)
    report['complete'] = True
    report['all_selected_oracles_successful'] = all(row['success'] for row in report['oracle_replays'])
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not report['all_selected_oracles_successful']:
        raise RuntimeError('A transformed expert-label replay failed; retain the report for diagnosis')


if __name__ == '__main__':
    main()
