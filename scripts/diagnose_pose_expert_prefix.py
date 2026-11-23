"""Compare reference and SFT after an identical expert prefix, never as autonomy."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from diagnose_pose_imitation import pin_training_reset
from evaluate_vlabench_reference import load_upstream_adapter
from pi05_runtime import file_sha256, restore_trainable
from reference_pose_lerobot import IMAGE_KEYS, load_pose_split
from select_fruit import video
from train_reference_aligned_fruit import matched_rollout
from train_reference_aligned_fruit import load_aligned_policy


class ExpertPrefix:
    def __init__(self, policy, episode, controls):
        self.policy, self.episode, self.controls = policy, episode, controls
        self.expert_controls = 0
        self.policy_calls = 0
        self.takeover = None
        self.prefix_state_max_error = 0.0

    def infer(self, observation):
        if self.expert_controls < self.controls:
            frame = self.expert_controls
            self.prefix_state_max_error = max(self.prefix_state_max_error, float(np.max(np.abs(
                observation['observation/state'] - self.episode.state[frame]))))
            self.expert_controls += 5
            return {'actions': self.episode.chunks[frame].copy()}
        if self.takeover is None:
            images = hashlib.sha256()
            for key in IMAGE_KEYS:
                images.update(key.encode())
                images.update(np.ascontiguousarray(observation[key]).tobytes())
            self.takeover = dict(
                measured_state=np.asarray(observation['observation/state']).tolist(),
                state_sha256=hashlib.sha256(np.ascontiguousarray(
                    observation['observation/state']).tobytes()).hexdigest(),
                rgb_sha256=images.hexdigest(), instruction=observation['prompt'])
        self.policy_calls += 1
        # Expert actions, rewards and target geometry are excluded from model input.
        assert set(observation) == {*IMAGE_KEYS, 'prompt', 'observation/state'}
        return self.policy.infer(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'run', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prefix-controls', type=int, default=140)
    args = parser.parse_args()
    if args.output.exists() or args.prefix_controls < 5 or args.prefix_controls % 5:
        parser.error('Use a fresh output directory and a positive five-control prefix')
    episodes = [ep for ep in load_pose_split(args.dataset, 'train')
                if (ep.task, ep.seed) == ('insert_flower', 42)]
    episode, = episodes
    if not args.prefix_controls < len(episode.state) or episode.policy_prefix_steps:
        parser.error('The prefix must end before this complete training demonstration')
    args.output.mkdir(parents=True)
    expected = pin_training_reset(episode, 0)[episode.seed]
    policy = load_aligned_policy(args.run, args.base, args.fork, stage='base')
    checkpoint = torch.load(args.run / 'sft.pt', map_location='cpu', weights_only=True)
    if checkpoint['metadata']['render_samples'] != 0:
        raise ValueError('The selected SFT checkpoint uses a different render profile')
    adapter = load_upstream_adapter(args.fork)
    report = dict(kind='expert_prefix_assisted_diagnostic_not_autonomous_success',
                  source_sha256=file_sha256(__file__), sft_sha256=file_sha256(args.run / 'sft.pt'),
                  dataset_verification_sha256=file_sha256(args.dataset / 'verification.json'),
                  prefix_controls=args.prefix_controls, max_total_controls=600, control_hz=10,
                  execute_controls=5, optimizer_updates=0,
                  sampler_rule='Reset seed 42 and consume one noise chunk per five expert controls before takeover.',
                  complete=False, stages={})
    for stage in ('reference', 'sft'):
        if stage == 'sft':
            restore_trainable(policy.model, checkpoint['policy'])
        policy.reset_rng(episode.seed)
        for _ in range(args.prefix_controls // 5):
            assert policy.reference_sample_noise() is not None
        client = ExpertPrefix(policy, episode, args.prefix_controls)
        trace_path = args.output / f'{stage}-controls.jsonl'
        started = time.monotonic()
        with trace_path.open('x') as trace, matched_rollout(
                episode.seed, episode.task, 'official', adapter(client, replan_steps=5),
                expected, 600, True, trace, render_samples=0) as (row, replay):
            video(args.output / f'{stage}.mp4', replay, 10,
                  f'Expert prefix {args.prefix_controls} controls + {stage} model | NOT autonomous',
                  episode.prompt)
        if client.expert_controls != args.prefix_controls or not client.policy_calls:
            raise RuntimeError('The model did not actually take over from the expert prefix')
        if stage == 'sft' and client.takeover != report['stages']['reference']['takeover']:
            raise RuntimeError('Reference and SFT takeover observations differ')
        report['stages'][stage] = dict(row, expert_controls=client.expert_controls,
            adapter_calls=row['policy_calls'], policy_calls=client.policy_calls,
            prefix_state_max_error=client.prefix_state_max_error,
            takeover=client.takeover, wall_seconds=time.monotonic() - started,
            counts_as_autonomous_policy_success=False,
            control_trace_sha256=file_sha256(trace_path))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage, success=row['success'], steps=row['steps'],
                              expert_controls=client.expert_controls,
                              policy_calls=client.policy_calls,
                              takeover=client.takeover)), flush=True)
    report['complete'] = True
    report['interpretation_limit'] = (
        'Matched expert-prefix takeover can test a near-demonstration suffix. '
        'Its outcomes are neither autonomous success rates nor proof of a unique training cause.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
