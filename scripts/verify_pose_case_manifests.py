"""Check frozen pose reset cases without loading a model or executing actions."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(cases, dataset_file, protocol_file):
    dataset = json.loads(dataset_file.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_file.read_text(encoding='utf-8'))
    if dataset['profile'] != 'vlabench_reference_pose' or not all(
            split['all_frames_verified'] for split in dataset['splits'].values()):
        raise ValueError('Expected an entirely verified pose dataset')
    sources = [row for split in dataset['splits'].values() for row in split['episodes_detail']]
    source_hashes = {row['initial_state_sha256'] for row in sources}
    source_ids = {(row['task'], row['seed']) for row in sources}
    files, suites = {}, {}
    for suite_name in ('ordinary', 'fruit_language'):
        suite = protocol[suite_name]
        low, high = suite['seeds_inclusive']
        groups = {}
        for task in suite['tasks']:
            seeds = suite.get('seeds_by_task', {}).get(task, list(range(low, high + 1)))
            if len(seeds) != len(set(seeds)):
                raise ValueError('Duplicate protocol seeds')
            for mode in suite.get('prompt_modes', ['official']):
                suffix = 'language-official' if suite_name == 'fruit_language' and mode == 'official' else mode
                path = cases / f'{task}-{suffix}.jsonl'
                rows = read_rows(path)
                if len(rows) != len(seeds) or {row['seed'] for row in rows} != set(seeds):
                    raise ValueError(f'Missing, extra, or duplicate cases in {path}')
                if len({row['initial_state_sha256'] for row in rows}) != len(rows):
                    raise ValueError(f'Repeated initial physics in {path}')
                for row in rows:
                    if (row['task'] != task or row['prompt_mode'] != mode
                            or row['render_samples'] != protocol['render_samples']
                            or row['initial_state_sha256'] in source_hashes
                            or (task, row['seed']) in source_ids):
                        raise ValueError(f'Incompatible or training-overlapping case in {path}')
                    if any(len(row[key]) != 64 for key in ('initial_state_sha256', 'initial_policy_rgb_sha256')):
                        raise ValueError(f'Missing reset hashes in {path}')
                    if any(key in row for key in ('success', 'action_sha256', 'steps')):
                        raise ValueError('Reset manifests must not contain policy performance')
                groups[task, mode] = {row['seed']: row for row in rows}
                files[path.name] = dict(cases=len(rows), sha256=sha256(path))
        if suite_name == 'fruit_language':
            for task in suite['tasks']:
                official = groups[task, 'official']
                for mode in ('paraphrase', 'alternate_target'):
                    for seed, changed in groups[task, mode].items():
                        original = official[seed]
                        for key in ('initial_state_sha256', 'initial_policy_rgb_sha256',
                                    'camera_names', 'model_render_options'):
                            if changed[key] != original[key]:
                                raise ValueError(f'Language cases changed {key} for {task}/{seed}')
                        if changed['instruction'] == original['instruction']:
                            raise ValueError('Language ablation did not change the instruction')
        suites[suite_name] = sum(len(rows) for rows in groups.values())
    invalid = []
    for task in protocol['ordinary']['tasks']:
        path = cases / f'{task}-invalid-resets.jsonl'
        if not path.exists():
            continue
        rows = read_rows(path)
        if any(row['reason'] != 'reset_physics_error_before_policy_execution'
               or row['attempts'] != len(row['errors']) for row in rows):
            raise ValueError(f'Invalid reset audit in {path}')
        invalid.extend((row['task'], row['seed']) for row in rows)
        files[path.name] = dict(rejected_resets=len(rows), sha256=sha256(path))
    expected_invalid = {(row['task'], row['seed']) for row in protocol['ordinary'].get('excluded_initializations', [])}
    if len(invalid) != len(set(invalid)) or set(invalid) != expected_invalid:
        raise ValueError('Rejected reset audit does not match the frozen protocol')
    return dict(protocol_sha256=sha256(protocol_file), dataset_verification_sha256=sha256(dataset_file),
                source_episodes=len(sources), suites=suites, files=files,
                excluded_initializations=[dict(task=task, seed=seed) for task, seed in sorted(invalid)],
                disjoint_from_training_and_validation=True, language_initial_physics_and_images_match=True,
                policy_rollouts_executed=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'dataset-verification', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.cases, args.dataset_verification, args.protocol)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
