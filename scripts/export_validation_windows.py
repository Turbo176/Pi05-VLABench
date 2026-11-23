"""Sample aligned 10-control windows from a verified LeRobot v2.1 validation split."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VIEWS = ('forward', 'wrist')
COLUMNS = ['frame_index', 'task_index', 'observation.state', 'action',
           *(f'observation.images.{name}' for name in VIEWS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.manifest.read_text(encoding='utf-8'))['splits']['validation']
    info = json.loads((args.validation / 'meta/info.json').read_text(encoding='utf-8'))
    if args.validation.name != 'validation' or not report['all_frames_verified'] or info['codebase_version'] != 'v2.1':
        raise ValueError('Use a verified LeRobot v2.1 validation split')
    tasks = {row['task_index']: row['task'] for row in map(json.loads,
             (args.validation / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines())}
    rows = []
    for episode, expected in enumerate(report['episodes_detail']):
        filename = info['data_path'].format(episode_chunk=episode // info['chunks_size'], episode_index=episode)
        data = pq.read_table(args.validation / filename, columns=COLUMNS)
        if len(data) != expected['frames'] or len(data) < 10:
            raise ValueError(f'Unexpected episode length in {filename}')
        np.testing.assert_array_equal(data['frame_index'].to_numpy(), np.arange(len(data)))
        for index in np.unique(np.linspace(0, len(data) - 10, 16, dtype=int)).tolist():
            prompt = tasks[data['task_index'][index].as_py()]
            if prompt != expected['instruction']:
                raise ValueError(f'Instruction mismatch in {filename}')
            images = {name: data[f'observation.images.{name}'][index].as_py()['bytes'] for name in VIEWS}
            rows.append({'task_id': expected['task'], 'prompt': prompt, 'frame': index,
                         'state': data['observation.state'][index].as_py(),
                         'actions': data['action'][index:index + 10].to_pylist(), **images})
    with args.output.open('xb') as stream:
        pq.write_table(pa.Table.from_pylist(rows), stream, compression='zstd')
    print(f'Wrote {len(rows)} windows from {len(report["episodes_detail"])} episodes')


if __name__ == '__main__':
    main()
