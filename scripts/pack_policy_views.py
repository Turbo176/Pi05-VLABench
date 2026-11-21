"""Transfer the two policy cameras losslessly and reconstruct the SFT input columns."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from pi05_data import COLUMNS, VIEWS

HEIGHT = 256
WIDTH = 256
FRAME_BYTES = HEIGHT * WIDTH * len(VIEWS) * 3


def ffmpeg(*args):
    return [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', *args]


def verify_frame_bytes(pipe, rows, fn):
    digest = hashlib.sha256()
    for i in range(rows):
        frame = pipe.read(FRAME_BYTES)
        if len(frame) != FRAME_BYTES:
            raise ValueError(f'Truncated raw RGB at frame {i}')
        digest.update(frame)
        fn(frame)
    if pipe.read(1):
        raise ValueError('Unexpected additional RGB frames')
    return digest.hexdigest()


def decode(video, rows, fn):
    with subprocess.Popen(ffmpeg('-threads', '2', '-i', str(video), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'),
                          stdout=subprocess.PIPE) as process:
        try:
            checksum = verify_frame_bytes(process.stdout, rows, fn)
        finally:
            process.stdout.close()
        if process.wait():
            raise RuntimeError(f'FFmpeg decode failed for {video}')
    return checksum


def pack_episode(train, archive, entry):
    path = train / entry
    stem = archive / entry.stem
    video = stem.with_suffix('.mkv')
    vectors = [name for name in COLUMNS if name not in VIEWS]
    table = pq.ParquetFile(path).read(columns=COLUMNS)
    pq.write_table(table.select(vectors).replace_schema_metadata(None), stem.with_suffix('.parquet'))
    digest = hashlib.sha256()
    with subprocess.Popen(ffmpeg('-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{WIDTH * len(VIEWS)}x{HEIGHT}',
                                 '-framerate', '10', '-i', '-', '-c:v', 'libx264rgb', '-preset', 'fast',
                                 '-crf', '0', '-threads', '2', str(video)), stdin=subprocess.PIPE) as process:
        try:
            for i in range(len(table)):
                images = []
                for name in VIEWS:
                    with Image.open(io.BytesIO(table[name][i].as_py()['bytes'])) as img:
                        images.append(np.asarray(img.convert('RGB')))
                frame = np.concatenate(images, axis=1).tobytes()
                if len(frame) != FRAME_BYTES:
                    raise ValueError(f'Invalid image shape in {entry}, frame {i}')
                digest.update(frame)
                process.stdin.write(frame)
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError(f'FFmpeg encode failed for {entry}')
    if decode(video, len(table), lambda frame: None) != digest.hexdigest():
        raise RuntimeError(f'Lossless RGB round trip failed for {entry}')
    print(entry, len(table), video.stat().st_size, flush=True)
    return {'file': entry.as_posix(), 'frames': len(table), 'rgb_sha256': digest.hexdigest()}


def unpack_episode(archive, train, manifest):
    entry = Path(manifest['file'])
    if len(entry.parts) != 3 or entry.parts[0] != 'data' or any(part == '..' for part in entry.parts):
        raise ValueError(f'Invalid manifest path {entry}')
    stem = archive / entry.stem
    table = pq.ParquetFile(stem.with_suffix('.parquet')).read()
    if manifest['frames'] != len(table):
        raise ValueError(f'Frame count mismatch in {entry}')
    cameras = {name: [] for name in VIEWS}

    def collect(frame):
        pixels = np.frombuffer(frame, np.uint8).reshape(HEIGHT, WIDTH * len(VIEWS), 3)
        for i, name in enumerate(VIEWS):
            with io.BytesIO() as buffer:
                Image.fromarray(pixels[:, i * WIDTH:(i + 1) * WIDTH]).save(buffer, format='PNG', compress_level=1)
                cameras[name].append({'bytes': buffer.getvalue(), 'path': None})

    if decode(stem.with_suffix('.mkv'), len(table), collect) != manifest['rgb_sha256']:
        raise ValueError(f'RGB checksum mismatch in {entry}')
    image_type = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    for name, values in cameras.items():
        table = table.append_column(name, pa.array(values, type=image_type))
    path = train / entry
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(entry, len(table), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pack', 'unpack'))
    parser.add_argument('train', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    with ThreadPoolExecutor(args.jobs) as pool:
        if args.mode == 'pack':
            args.archive.mkdir(parents=True, exist_ok=False)
            shutil.copytree(args.train / 'meta', args.archive / 'meta')
            paths = sorted(p.relative_to(args.train) for p in (args.train / 'data').glob('*/episode_*.parquet'))
            if len(paths) != json.loads((args.train / 'meta/info.json').read_text())['total_episodes']:
                raise ValueError('Training files differ from LeRobot metadata')
            manifest = list(pool.map(lambda p: pack_episode(args.train, args.archive, p), paths))
            (args.archive / 'transport.json').write_text(json.dumps(manifest, indent=2) + '\n')
        else:
            manifest = json.loads((args.archive / 'transport.json').read_text())
            args.train.mkdir(parents=True, exist_ok=False)
            shutil.copytree(args.archive / 'meta', args.train / 'meta')
            list(pool.map(lambda entry: unpack_episode(args.archive, args.train, entry), manifest))


if __name__ == '__main__':
    main()
