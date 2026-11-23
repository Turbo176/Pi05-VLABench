"""Localize a matched early/late checkpoint regression by swapping parameter groups."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from diagnose_pose_imitation import pin_training_reset
from pi05_runtime import ACTOR_MODULES, file_sha256, restore_trainable
from reference_pose_lerobot import load_pose_split
from train_reference_aligned_fruit import evaluate, load_aligned_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'early-run', 'late-run', 'early-gate', 'late-gate', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--case', required=True, metavar='TASK:SEED')
    args = parser.parse_args()
    task, seed_text = args.case.rsplit(':', 1)
    seed = int(seed_text)
    episode, = [ep for ep in load_pose_split(args.dataset, 'train')
                if (ep.task, ep.seed) == (task, seed)]
    early = torch.load(args.early_run / 'sft.pt', map_location='cpu', weights_only=True)
    late = torch.load(args.late_run / 'sft.pt', map_location='cpu', weights_only=True)
    for key in ('base_sha256', 'norm_stats_sha256', 'seed', 'diagnostic_case', 'training_scope',
                'train_action_heads', 'actor_lora_rank', 'vlm_lora_rank', 'render_samples',
                'shortest_euler', 'discrete_state', 'horizon', 'execute', 'lr'):
        if early['metadata'].get(key) != late['metadata'].get(key):
            raise ValueError(f'The compared checkpoints differ in {key}')
    if (early['metadata']['updates'] >= late['metadata']['updates']
            or early['metadata']['vlm_lora_rank'] != 0
            or not early['metadata']['train_action_heads']
            or set(early['policy']) != set(late['policy'])
            or (args.early_run / 'norm_stats.json').read_bytes()
               != (args.late_run / 'norm_stats.json').read_bytes()):
        raise ValueError('Require ordered checkpoints with identical interfaces and frozen VLM')
    gates = {}
    for name, path, run in (('early', args.early_gate, args.early_run),
                            ('late', args.late_gate, args.late_run)):
        gate = json.loads(path.read_text(encoding='utf-8'))
        if (not gate['complete'] or gate['sft_sha256'] != file_sha256(run / 'sft.pt')
                or (gate['task'], gate['seed']) != (task, seed)
                or gate['expert_actions_supplied_to_policy'] or gate['optimizer_updates']):
            raise ValueError('Require completed real autonomous comparison records')
        gates[name] = dict(source_sha256=file_sha256(path), sft_sha256=gate['sft_sha256'],
                           passed=gate['passed'], successes=gate['successes'],
                           completed_repeats=gate['completed_repeats'])
    if not gates['early']['passed'] or gates['late']['passed']:
        raise ValueError('This diagnostic requires an observed early success and later failure')
    groups = {
        'action_heads': [name for name in early['policy'] if name.startswith(ACTOR_MODULES)],
        'action_adapters': [name for name in early['policy'] if not name.startswith(ACTOR_MODULES)],
    }
    if any(not names for names in groups.values()):
        raise ValueError('Both compared parameter groups must be present')
    deltas = {}
    for group, names in groups.items():
        total_squared, maximum, elements = 0.0, 0.0, 0
        for name in names:
            difference = late['policy'][name].float() - early['policy'][name].float()
            total_squared += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            elements += difference.numel()
        deltas[group] = dict(tensors=len(names), elements=elements,
                             change_l2=total_squared ** 0.5, max_abs_change=maximum)
    expected = pin_training_reset(episode, episode.render_samples)
    policy = load_aligned_policy(args.early_run, args.base, args.fork)
    policy.provenance.pop('sft_sha256')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='parameter_group_intervention_not_a_trained_policy_result',
                  case=args.case, optimizer_updates=0, expert_actions_supplied_to_policy=False,
                  script_sha256=file_sha256(__file__), source_gates=gates,
                  parameter_group_differences=deltas, interventions=[], complete=False)
    destination = args.output / 'audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for name, head_source, adapter_source in (
            ('early_heads_late_adapters', early, late),
            ('late_heads_early_adapters', late, early)):
        weights = {key: (head_source if key.startswith(ACTOR_MODULES) else adapter_source)['policy'][key]
                   for key in early['policy']}
        restore_trainable(policy.model, weights)
        digest = hashlib.sha256()
        for key in sorted(weights):
            digest.update(key.encode('utf-8'))
            digest.update(weights[key].contiguous().view(torch.uint8).numpy().tobytes())
        policy.provenance.update(diagnostic_intervention=name,
                                 hybrid_parameters_sha256=digest.hexdigest(),
                                 source_early_sft_sha256=gates['early']['sft_sha256'],
                                 source_late_sft_sha256=gates['late']['sft_sha256'])
        out = args.output / name
        evaluate(policy, out, expected, 600, False, 'sft', seeds=[seed], trace=True,
                 task=task, render_samples=episode.render_samples)
        row, = map(json.loads, (out / 'episodes.jsonl').read_text(encoding='utf-8').splitlines())
        report['interventions'].append(dict(name=name, result=row))
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    report['complete'] = True
    report['interpretation_limit'] = (
        'Each hybrid executes once from this fixed seen reset and sampler. '
        'The swaps isolate parameter-group interactions here; they do not establish '
        'generalization, a universal cause, or improvement of a normally trained checkpoint.')
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
