"""Freeze fruit-only experiment inputs before opening reserved policy outcomes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    project = args.project_root.resolve()
    code = Path(__file__).resolve().parents[1]
    training_path = code / 'configs/ppo-fruit-completion-v1.json'
    protocol_path = code / 'configs/evaluation-fruit-completion-v1.json'
    training = json.loads(training_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    parent = project / training['parent_run']
    actual_parent = sha256(parent / 'sft.pt')
    if actual_parent != training['parent_sft_sha256'] or actual_parent != protocol['sft_sha256']:
        raise ValueError('The frozen SFT parent differs from the registered checkpoint')
    output = project / 'runs/pose-robustness-v2/fruit-completion-v1'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'registration.json').exists():
        raise FileExistsError('This experiment has already been registered')
    cases = output / 'cases'
    cases.mkdir(exist_ok=False)
    sources = project / protocol['case_manifest_source']
    selected = {}
    for suite_name in ('ordinary', 'fruit_language'):
        suite = protocol[suite_name]
        seeds = set(range(suite['seeds_inclusive'][0], suite['seeds_inclusive'][1] + 1))
        for mode in suite.get('prompt_modes', ['official']):
            stem = ('select_fruit-language-official' if suite_name == 'fruit_language' and mode == 'official'
                    else 'select_fruit-' + mode)
            source = sources / (stem + '.jsonl')
            lines = [line for line in source.read_text(encoding='utf-8').splitlines()
                     if json.loads(line)['seed'] in seeds]
            rows = [json.loads(line) for line in lines]
            if len(rows) != len(seeds) or {r['seed'] for r in rows} != seeds:
                raise ValueError('Incomplete or duplicated initialization cases: ' + stem)
            if any(r['task'] != 'select_fruit' or r['prompt_mode'] != mode
                   or r['render_samples'] != protocol['render_samples'] for r in rows):
                raise ValueError('Unexpected initialization interface: ' + stem)
            destination = cases / (stem + '.jsonl')
            destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            selected[stem] = dict(seeds=sorted(seeds), source_sha256=sha256(source),
                                  subset_sha256=sha256(destination))
    inputs = [training_path, protocol_path]
    inputs += [code / 'scripts' / name for name in (
        'register_fruit_completion.py', 'check_reference_rl_handoff.py',
        'ppo_reference_aligned_fruit.py', 'ppo_pi05.py', 'pi05_runtime.py',
        'train_reference_aligned_fruit.py', 'evaluate_pose_development.py',
        'evaluate_vlabench_reference.py', 'select_fruit.py')]
    snapshots = output / 'sources'
    snapshots.mkdir(exist_ok=False)
    hashes = {}
    for source in inputs:
        relative = source.relative_to(code)
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(relative)] = sha256(source)
    report = dict(registered_at=datetime.now(timezone.utc).isoformat(),
                  sft_sha256=actual_parent, norm_stats_sha256=sha256(parent / 'norm_stats.json'),
                  inputs=hashes, manifests=selected, reserved_policy_outcomes_consulted=False,
                  rule='Single fixed PPO descendant; no outcome-dependent tuning or case exclusions.')
    with (output / 'registration.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
