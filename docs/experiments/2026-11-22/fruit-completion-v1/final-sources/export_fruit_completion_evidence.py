"""Bundle completed fruit experiment evidence without data, assets or model weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root.parents[2]
    code = project / 'code'
    if (root / 'exit-status.txt').read_text().strip() != '0':
        raise ValueError('Quantitative pipeline is incomplete')
    if (root / 'artifacts-exit-status.txt').read_text().strip() != '0':
        raise ValueError('Comparison and recorded-video verification are incomplete')
    comparison = json.loads((root / 'comparison.json').read_text())
    files = {}
    allowed = {'.json', '.jsonl', '.txt', '.log', '.py', '.sh', '.png', '.svg', '.jpg'}
    for source in sorted(root.rglob('*')):
        if source.is_file() and not source.is_symlink() and source.suffix in allowed:
            relative = source.relative_to(root)
            if relative.name == 'evidence-export.json':
                continue
            files['evidence/' + relative.as_posix()] = source
    parent = project / 'runs/pose-robustness-v2/two-task-conservative-90'
    for name in ('summary.json', 'updates.jsonl', 'norm_stats.json'):
        files['evidence/sft-parent/' + name] = parent / name
    files['evidence/development-reference.jsonl'] = project / 'runs/pose-robustness-v2/base-development/select_fruit/episodes.jsonl'
    files['evidence/development-sft.jsonl'] = project / 'runs/pose-robustness-v2/two-task-conservative-development/episodes.jsonl'
    for name in ('summarize_fruit_completion.py', 'plot_fruit_completion.py', 'record_fruit_completion.py',
                 'preview_fruit_completion.py',
                 'export_fruit_completion_evidence.py', 'finalize_fruit_completion_artifacts.py',
                 'run-fruit-completion-v1.sh',
                 'run-fruit-completion-artifacts-v1.sh'):
        files['evidence/final-sources/' + name] = code / 'scripts' / name
    for name in ('evaluation-fruit-completion-v1.json', 'ppo-fruit-completion-v1.json'):
        files['evidence/final-configs/' + name] = code / 'configs' / name
    for stage in ('base', 'sft', 'ppo'):
        verification = json.loads((root / ('video-' + stage) / 'verification.json').read_text())
        source = root / ('video-' + stage) / 'select_fruit-1501.mp4'
        if file_hash(source) != verification['video_sha256']:
            raise ValueError('Video changed after verification: ' + stage)
        files['media/fruit-completion-v1-' + stage + '-1501.mp4'] = source
        files['media/fruit-completion-v1-' + stage + '-1501.gif'] = root / 'previews' / ('fruit-completion-v1-' + stage + '-1501.gif')
        files['evidence/video-' + stage + '/observed-frames.jpg'] = source.parent / 'observed-frames.jpg'
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    scope=comparison['scope'], sft_sha256=comparison['sft_sha256'],
                    ppo_sha256=comparison['ppo_sha256'],
                    files=[dict(bundle_path=name, source=str(path), bytes=path.stat().st_size,
                                sha256=file_hash(path)) for name, path in sorted(files.items())])
    manifest_path = root / 'evidence-export.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    archive_path = root / 'evidence-export.tar.gz'
    with tarfile.open(archive_path, 'x:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
        archive.add(manifest_path, arcname='evidence/evidence-export.json', recursive=False)
    print(json.dumps(dict(archive=str(archive_path), bytes=archive_path.stat().st_size,
                          sha256=file_hash(archive_path), exported_files=len(files))), flush=True)


if __name__ == '__main__':
    main()
