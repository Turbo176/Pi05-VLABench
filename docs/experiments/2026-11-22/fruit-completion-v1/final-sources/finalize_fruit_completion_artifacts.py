"""Recheck scored fruit artifacts after recovering the confidence-interval plot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fruit_case(path, seed):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    selected = [row for row in rows if row['task'] == 'select_fruit'
                and row['seed'] == seed and row['prompt_mode'] == 'official']
    if len(selected) != 1:
        raise ValueError('Expected one scored fruit case: ' + str(path))
    return selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root.parents[2]
    if (root / 'exit-status.txt').read_text().strip() != '0':
        raise ValueError('Quantitative pipeline is incomplete')
    comparison_path = root / 'comparison.json'
    comparison = read_json(comparison_path)
    for name, item in comparison['inputs'].items():
        if digest(project / item['path']) != item['sha256']:
            raise ValueError('Scored records changed: ' + name)
    plot = read_json(root / 'figures/plot-report.json')
    if plot['comparison_sha256'] != digest(comparison_path):
        raise ValueError('Plot used a different comparison')
    if plot['source_sha256'] != digest(project / 'code/scripts/plot_fruit_completion.py'):
        raise ValueError('Plot source changed')
    if len(plot['figures']) != 2:
        raise ValueError('Expected PNG and SVG figures')
    for item in plot['figures']:
        if digest(project / item['path']) != item['sha256']:
            raise ValueError('Figure changed: ' + item['path'])
    seed = read_json(root / 'video-plan.json')['seed']
    videos = {}
    for stage in ('base', 'sft', 'ppo'):
        output = root / ('video-' + stage)
        verification = read_json(output / 'verification.json')
        expected = fruit_case(project / comparison['inputs']['ordinary/' + stage]['path'], seed)
        actual = fruit_case(output / 'episodes.jsonl', seed)
        fields = set(comparison['paired_interface_fields']) | {
            'success', 'steps', 'action_sha256', 'first_action_sha256', 'ik_nonconverged_steps'}
        if stage != 'base':
            fields.add('sft_sha256')
        if stage == 'ppo':
            fields.add('ppo_sha256')
        for field in fields:
            if actual[field] != expected[field]:
                raise ValueError('Recorded rollout differs: ' + stage + '/' + field)
        video = output / ('select_fruit-' + str(seed) + '.mp4')
        if (digest(video) != verification['video_sha256']
                or video.stat().st_size != verification['video_bytes']
                or actual['success'] != verification['success']
                or actual['steps'] != verification['controls']):
            raise ValueError('Video verification changed: ' + stage)
        source_hash = verification['source_sha256']
        source = project / 'code/scripts/record_fruit_completion.py'
        if digest(source) != source_hash:
            source = root / 'source-snapshots' / source_hash / 'record_fruit_completion.py'
        if digest(source) != source_hash:
            raise ValueError('Executed recorder source is unavailable: ' + stage)
        preview = root / 'previews' / ('fruit-completion-v1-' + stage + '-' + str(seed) + '.gif')
        if not preview.is_file() or preview.stat().st_size == 0:
            raise ValueError('Missing preview: ' + stage)
        videos[stage] = dict(success=actual['success'], controls=actual['steps'],
                             matched_fields=sorted(fields), video_sha256=digest(video),
                             preview_sha256=digest(preview), recorder_source_sha256=source_hash)
    status = root / 'artifacts-exit-status.txt'
    previous = status.read_text().strip()
    saved_status = root / 'artifacts-before-plot-fix-exit-status.txt'
    if not saved_status.exists():
        shutil.copyfile(status, saved_status)
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(),
                  reason='Recover plotting-only failure: Wilson lower bound rounded slightly above zero at zero successes.',
                  change='Clamp plotted error lengths to zero; scored records and reported intervals are unchanged.',
                  previous_artifact_exit_status=previous, comparison_sha256=digest(comparison_path),
                  verified_scored_inputs=len(comparison['inputs']), figures=plot['figures'], videos=videos,
                  source_sha256=digest(Path(__file__)))
    with (root / 'artifacts-recovery.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    status.write_text('0\n', encoding='utf-8')
    (root / 'artifacts-recovered-at.txt').write_text(report['completed_at'] + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
