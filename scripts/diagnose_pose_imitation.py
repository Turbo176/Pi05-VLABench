"""Compare reference and SFT actions on recorded expert observations and training resets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_vlabench_reference import build_episode
from pi05_runtime import file_sha256, restore_trainable
from reference_pose_lerobot import load_pose_split
from select_fruit import open_episode
from train_reference_aligned_fruit import EXECUTE, evaluate, load_aligned_policy, wrap_euler


def action_metrics(prediction, target):
    error = prediction - target
    error[:, 3:6] = wrap_euler(error[:, 3:6])
    open_prediction, open_target = prediction[:, 6] >= 0.1, target[:, 6] >= 0.1
    return dict(
        controls=len(target), xyz_mae_m=float(np.abs(error[:, :3]).mean()),
        xyz_distance_median_m=float(np.median(np.linalg.norm(error[:, :3], axis=1))),
        xyz_distance_p95_m=float(np.quantile(np.linalg.norm(error[:, :3], axis=1), 0.95)),
        euler_mae_rad=float(np.abs(error[:, 3:6]).mean()),
        gripper_mae=float(np.abs(error[:, 6]).mean()),
        gripper_command_agreement=float(np.mean(open_prediction == open_target)),
        expert_open_controls=int(open_target.sum()),
        predicted_closed_when_expert_open=int((~open_prediction & open_target).sum()),
        predicted_open_when_expert_closed=int((open_prediction & ~open_target).sum()),
    )


def teacher_forced(policy, episode, stride):
    predictions, targets, windows = [], [], []
    chunks_before = episode.chunks.copy()
    for frame in range(0, len(episode.state), stride):
        row = episode.row(frame)
        # Keep labels out of inference, just as in actual policy rollouts.
        target = row.pop('actions').copy()[:EXECUTE]
        valid = min(EXECUTE, len(episode.state) - frame)
        policy.reset_rng(100000 + episode.seed * 1000 + frame)
        prediction = policy.infer(row)['actions'][:valid].copy()
        target = target[:valid]
        predictions.append(prediction)
        targets.append(target)
        windows.append(dict(frame=frame, **action_metrics(prediction, target)))
    np.testing.assert_array_equal(episode.chunks, chunks_before)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    groups = {}
    for name, mask in (('expert_open', target[:, 6] >= 0.1),
                       ('expert_closed', target[:, 6] < 0.1)):
        if mask.any():
            groups[name] = action_metrics(prediction[mask], target[mask])
    return dict(task=episode.task, seed=episode.seed, instruction=episode.prompt,
                kind='teacher_forced_expert_observations_not_policy_rollout',
                stride=stride, padded_terminal_controls_excluded=True,
                metrics=action_metrics(prediction, target), by_gripper_label=groups,
                per_window=windows)


def pin_training_reset(episode, render_samples):
    with open_episode(episode.seed, episode.task, render_samples=render_samples) as env:
        row, rollout = build_episode(env, None, episode.seed, 1, False, task=episode.task)
        rollout.close()
    if row['initial_state_sha256'] != episode.initial_state or row['instruction'] != episode.prompt:
        raise ValueError('The diagnostic reset differs from its training demonstration')
    return {episode.seed: row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'run', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--case', action='append', metavar='TASK:SEED', required=True)
    parser.add_argument('--rollout-task', action='append', default=[],
                        help='Also run the full policy on this case; it is a training-set diagnostic')
    parser.add_argument('--stride', type=int, default=EXECUTE)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.stride <= EXECUTE:
        parser.error('Use a stride between one and the five executed controls')
    episodes = load_pose_split(args.dataset, 'train')
    requested = [tuple(case.rsplit(':', 1)) for case in args.case]
    selected = []
    for task, seed in requested:
        matches = [episode for episode in episodes if (episode.task, episode.seed) == (task, int(seed))]
        if len(matches) != 1:
            parser.error(f'Expected one training demonstration for {task}:{seed}')
        selected.extend(matches)
    if len({(episode.task, episode.seed) for episode in selected}) != len(selected):
        parser.error('Diagnostic cases must be distinct')
    if set(args.rollout_task) - {episode.task for episode in selected}:
        parser.error('A rollout task must have a selected training case')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = load_aligned_policy(args.run, args.base, args.fork, stage='base')
    checkpoint = torch.load(args.run / 'sft.pt', weights_only=True, map_location='cpu')
    render_samples = checkpoint['metadata']['render_samples']
    report = dict(kind='training_set_imitation_diagnostic_not_generalization_evaluation',
                  dataset_verification_sha256=file_sha256(args.dataset / 'verification.json'),
                  sft_sha256=file_sha256(args.run / 'sft.pt'),
                  script_sha256=file_sha256(__file__),
                  render_samples=render_samples, stages={})
    resets = {(episode.task, episode.seed): pin_training_reset(episode, render_samples)
              for episode in selected if episode.task in args.rollout_task}
    for stage in ('base', 'sft'):
        if stage == 'sft':
            restore_trainable(policy.model, checkpoint['policy'])
            policy.provenance.pop('configuration_source_sha256', None)
            policy.provenance['sft_sha256'] = report['sft_sha256']
        metrics = [teacher_forced(policy, episode, args.stride) for episode in selected]
        report['stages'][stage] = metrics
        for row in metrics:
            print(json.dumps(dict(stage=stage, task=row['task'], seed=row['seed'],
                                  teacher_forced_metrics=row['metrics'])), flush=True)
        (args.output / 'teacher-forced.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        for episode in selected:
            if episode.task not in args.rollout_task:
                continue
            evaluate(policy, args.output / stage / f'{episode.task}-{episode.seed}',
                     resets[episode.task, episode.seed], 600, args.video, stage,
                     seeds=[episode.seed], trace=True, task=episode.task,
                     render_samples=render_samples)
    print(json.dumps(dict(status='complete', output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
