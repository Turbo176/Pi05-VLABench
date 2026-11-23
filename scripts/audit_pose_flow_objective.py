"""Measure full-forward training versus cached inference and held-out action errors."""
import argparse
import json
from pathlib import Path

import torch
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation, _observation_to_dtype

from pi05_runtime import file_sha256, restore_trainable
from reference_pose_lerobot import load_pose_split
from train_reference_aligned_fruit import load_aligned_policy, validation_mae


def compare_velocity(policy, raw, time):
    observation, labels, _ = policy.encode(raw)
    noise = torch.randn(labels.shape, device=labels.device,
                        generator=torch.Generator(device=labels.device).manual_seed(911))
    instant = torch.full((1,), time, device=labels.device)
    captured = []
    hook = policy.model.action_out_proj.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().clone()))
    try:
        policy.model.compute_loss(observation, labels, train=False, noise=noise, time=instant)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one training velocity projection')
    prepared = _observation_to_dtype(preprocess_observation(observation, train=False), policy.model.embed_dtype)
    _, mask, cache = policy.model.build_prefix_cache(prepared)
    x_t = instant[:, None, None] * noise + (1 - instant[:, None, None]) * labels
    hidden = policy.model.run_suffix(prepared, x_t, instant, cache, mask)
    cached = policy.model.velocity_from_suffix(hidden)
    error = (captured[0] - cached).abs()
    return dict(time=time, velocity_max_abs_error=float(error.max()),
                velocity_mean_abs_error=float(error.mean()),
                active_velocity_mean_abs_error=float(error[..., :7].mean()),
                active_velocity_mean_magnitude=float(cached[..., :7].abs().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'configuration-run', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, action='append', default=[])
    args = parser.parse_args()
    episodes = load_pose_split(args.dataset, 'validation')
    tasks = sorted({episode.task for episode in episodes})
    policy = load_aligned_policy(args.configuration_run, args.base, args.fork, stage='base')
    candidates = [('published_reference', None), *[(path.name, path) for path in args.candidate_run]]
    report = dict(dataset_verification_sha256=file_sha256(args.dataset / 'verification.json'),
                  policy_rollouts_executed=0, stages={})
    with torch.inference_mode():
        for name, path in candidates:
            if path is not None:
                checkpoint = torch.load(path / 'sft.pt', weights_only=True, map_location='cpu')
                restore_trainable(policy.model, checkpoint['policy'])
                del checkpoint
            metrics = {'per_task': {}, 'full_forward_vs_cached_inference': []}
            if path is not None:
                metrics['sft_sha256'] = file_sha256(path / 'sft.pt')
            for task in tasks:
                subset = [episode for episode in episodes if episode.task == task]
                metrics['per_task'][task] = validation_mae(policy, subset)
                for instant in (0.1, 0.5, 0.9):
                    metrics['full_forward_vs_cached_inference'].append(
                        dict(task=task, **compare_velocity(policy, subset[0].row(0), instant)))
            report['stages'][name] = metrics
            print(json.dumps(dict(stage=name, **metrics)), flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
