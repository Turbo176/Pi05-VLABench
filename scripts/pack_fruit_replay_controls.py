import argparse
import json
from pathlib import Path

import h5py

FIELDS = ('action', 'state', 'time', 'post_state', 'terminal', 'success',
          'initial_physics_state', 'initial_body_xpos', 'initial_body_xquat')


def main():
    parser = argparse.ArgumentParser(description='Pack exact replay controls from a verified LeRobot export.')
    parser.add_argument('--verification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    train = json.loads(args.verification.read_text(encoding='utf-8'))['splits']['train']
    if not train['all_frames_verified']:
        raise ValueError('Expected a verified training export')
    with h5py.File(args.output, 'x') as pack:
        for index, row in enumerate(train['episodes_detail']):
            if row['task'] != 'select_fruit':
                continue
            with h5py.File(row['source'], 'r') as source:
                if (source.attrs['seed'] != row['seed'] or source.attrs['instruction'] != row['instruction']
                        or source.attrs['task'] != row['task'] or len(source['action']) != row['frames']
                        or source['action'].dtype.itemsize != 8 or not source['success'][-1]):
                    raise ValueError(f'Source differs from the verified episode {index}')
                episode = pack.create_group(str(row['seed']))
                episode.attrs.update(source.attrs)
                episode.attrs.update(lerobot_episode_index=index, initial_state_sha256=row['initial_state_sha256'])
                for field in FIELDS:
                    episode.create_dataset(field, data=source[field][:], compression='lzf')
        print(json.dumps({'episodes': len(pack), 'seeds': sorted(map(int, pack))}))


if __name__ == '__main__':
    main()
