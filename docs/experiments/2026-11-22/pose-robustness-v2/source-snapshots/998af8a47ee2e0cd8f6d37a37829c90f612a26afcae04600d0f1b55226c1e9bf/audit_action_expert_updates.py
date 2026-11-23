"""Measure stored action-expert parameter changes against the pinned parent."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch

from pi05_runtime import ACTOR_MODULES, file_sha256, is_action_expert_parameter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    path = args.run / 'sft.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    if not metadata.get('train_full_action_expert'):
        parser.error('This audit requires a declared full-action diagnostic checkpoint')
    base_sha = file_sha256(args.base)
    if base_sha != metadata['base_sha256']:
        raise ValueError('The audit parent differs from the training parent')
    groups, rows = {}, []
    with safe_open(str(args.base), framework='pt', device='cpu') as base:
        for name, value in checkpoint['policy'].items():
            expert = is_action_expert_parameter(name)
            if not expert and not name.startswith(ACTOR_MODULES):
                raise ValueError(f'Unexpected VLM or adapter weight in diagnostic: {name}')
            original = base.get_tensor(name).to(value.dtype)
            delta = value.float() - original.float()
            changed = int(torch.count_nonzero(delta))
            count = value.numel()
            group_name = ('action_expert_' if expert else 'action_heads_') + str(value.dtype)
            group = groups.setdefault(group_name, dict(parameters=0, changed_parameters=0,
                                                      absolute_delta_sum=0.0, max_absolute_delta=0.0))
            maximum = float(delta.abs().max())
            group['parameters'] += count
            group['changed_parameters'] += changed
            group['absolute_delta_sum'] += float(delta.abs().sum())
            group['max_absolute_delta'] = max(group['max_absolute_delta'], maximum)
            rows.append(dict(name=name, dtype=str(value.dtype), parameters=count,
                             changed_parameters=changed, max_absolute_delta=maximum))
    for group in groups.values():
        group['changed_fraction'] = group['changed_parameters'] / group['parameters']
        group['mean_absolute_delta'] = group.pop('absolute_delta_sum') / group['parameters']
    report = dict(kind='stored_parameter_change_not_autonomous_success',
                  script_sha256=file_sha256(__file__), checkpoint_sha256=file_sha256(path),
                  base_sha256=base_sha, groups=groups, tensors=rows,
                  interpretation='A nonzero gradient does not imply every BF16 value changed. '
                                 'These measurements describe this full-action diagnostic only; '
                                 'they do not identify the cause of prior LoRA failures.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == '__main__':
    main()
