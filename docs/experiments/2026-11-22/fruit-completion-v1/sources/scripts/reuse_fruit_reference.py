"""Reuse fixed reference records only after exact fresh-repeat and input audits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from openpi.shared import normalize
import torch

from train_reference_aligned_fruit import AlignedPolicy, FruitEpisode, paired_episodes


def compare_tree(left, right):
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('Reference transform keys differ')
        for key in left:
            compare_tree(left[key], right[key])
    else:
        np.testing.assert_array_equal(left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'old-run', 'run', 'source', 'fork', 'episode'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    old = torch.load(args.old_run / 'sft.pt', map_location='cpu', weights_only=True)['metadata']
    new = torch.load(args.run / 'sft.pt', map_location='cpu', weights_only=True)['metadata']
    required = ('base_sha256', 'norm_stats_sha256', 'horizon', 'execute', 'discrete_state',
                'reference_model_revision', 'actor_lora_rank', 'render_samples')
    for field in required:
        if old[field] != new[field]:
            raise ValueError('Reference inference configuration differs in ' + field)
    differences = {key: dict(existing=old.get(key), current=new.get(key))
                   for key in ('shortest_euler', 'vlm_lora_rank', 'train_action_heads')
                   if old.get(key) != new.get(key)}
    stats = normalize.deserialize_json((args.run / 'norm_stats.json').read_text())
    policies = [AlignedPolicy(None, stats, args.fork, shortest_euler=info['shortest_euler'],
                              discrete_state=info['discrete_state'], reference_noise=True)
                for info in (old, new)]
    episode = FruitEpisode(args.episode)
    for frame in (0, len(episode.state) // 2, len(episode.state) - 1):
        raw = episode.row(frame)
        raw.pop('actions')
        encoded = [policy.inputs(raw.copy()) for policy in policies]
        compare_tree(*encoded)
        action = np.zeros((10, 32), dtype=np.float32)
        compare_tree(*[policy.outputs(dict(actions=action.copy(), state=row['state']))
                       for policy, row in zip(policies, encoded)])
    existing = paired_episodes(args.source)
    old_config_sha256 = hashlib.sha256((args.old_run / 'sft.pt').read_bytes()).hexdigest()
    if any(row['configuration_source_sha256'] != old_config_sha256 for row in existing.values()):
        raise ValueError('The old reference records used a different configuration checkpoint')
    cases = paired_episodes(args.root / 'cases/select_fruit-official.jsonl')
    interface = ('task', 'seed', 'instruction', 'prompt_mode', 'initial_state_sha256',
                 'initial_policy_rgb_sha256', 'render_samples')
    for seed, case in cases.items():
        for field in interface:
            if existing[seed][field] != case[field]:
                raise ValueError('Existing reference reset differs: ' + str((seed, field)))
        expected = dict(base_sha256=new['base_sha256'], norm_stats_sha256=new['norm_stats_sha256'],
                        control_hz=10, replan_steps=5, max_control_steps=600,
                        sampler='JAX seeded noise', stage='base')
        if any(existing[seed][key] != value for key, value in expected.items()):
            raise ValueError('Existing reference used a different execution profile')
    repeated = []
    repeated_fields = (*interface, 'success', 'steps', 'first_action_sha256', 'action_sha256',
                       'ik_nonconverged_steps', 'stop_reason', 'failure_type')
    for path in sorted((args.root / 'ordinary-base').glob('cases/*/episodes.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for field in repeated_fields:
                if row[field] != existing[row['seed']][field]:
                    raise ValueError('Fresh reference repeat differs: ' + str((row['seed'], field)))
            repeated.append(row['seed'])
    if args.apply and not {1501, 1502, 1503} <= set(repeated):
        raise ValueError('Wait for all three first fresh repeats before reusing reference records')
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), source=str(args.source),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  retained_seeds=sorted(cases), exact_fresh_repeats=repeated,
                  repeated_fields=list(repeated_fields), inference_transform_windows=3,
                  configuration_differences=differences,
                  difference_explanation={
                      'shortest_euler': 'Only transforms action labels when actions are present; inference has no labels. Actual inference inputs and output transforms were compared exactly.',
                      'vlm_lora_rank': 'Base evaluation restores no trained adapters. VLM LoRALinear initializes its B matrix to zero; fresh complete action hashes must match.',
                      'train_action_heads': 'Changes requires_grad only; no SFT parameters or optimizer updates are loaded for base evaluation.'},
                  applied=args.apply,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.apply:
        output = args.root / 'ordinary-base'
        if (output / 'completed.json').exists():
            raise FileExistsError('The reference group is already complete')
        with (output / 'episodes.jsonl').open('x', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(existing[seed]) + '\n' for seed in sorted(cases)))
        (output / 'completed.json').write_text(json.dumps(dict(
            stage='base', suite='ordinary', cases=len(cases), successes=sum(existing[s]['success'] for s in cases),
            protocol_sha256=hashlib.sha256((args.root / 'sources/configs/evaluation-fruit-completion-v1.json').read_bytes()).hexdigest(),
            evidence_kind='Existing unchanged reference records with exact fresh-repeat verification',
            reference_reuse=report), indent=2) + '\n')
        (args.root / 'reference-reuse.json').write_text(json.dumps(report, indent=2) + '\n')
        shutil.copyfile(args.source, args.root / 'reference-original-20.jsonl')
        shutil.copyfile(__file__, args.root / 'sources/scripts/reuse_fruit_reference.py')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
