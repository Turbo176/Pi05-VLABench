"""Create labelled README previews and final-frame sheets from verified videos."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stages', choices=('base', 'sft', 'ppo'), nargs='+', default=['base', 'sft', 'ppo'])
    args = parser.parse_args()
    previews = args.root / 'previews'
    previews.mkdir(exist_ok=True)
    records = []
    for stage in args.stages:
        folder = args.root / ('video-' + stage)
        verification = json.loads((folder / 'verification.json').read_text())
        path = folder / 'select_fruit-1501.mp4'
        if hashlib.sha256(path.read_bytes()).hexdigest() != verification['video_sha256']:
            raise ValueError('A source video changed after its verified rollout')
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for fraction in (0, 1/3, 2/3, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round((count - 1) * fraction)))
            ok, image = capture.read()
            if not ok:
                raise ValueError('Cannot decode the final control frame')
            frames.append(image)
        capture.release()
        sheet = cv2.vconcat([cv2.hconcat(frames[:2]), cv2.hconcat(frames[2:])])
        sheet_path = folder / 'observed-final-frames.jpg'
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ValueError('Cannot write final-frame sheet')
        # Preview only: 2x playback. The MP4 remains the original 10 Hz recording.
        gif = previews / ('fruit-completion-v1-' + stage + '-1501.gif')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-n',
                   '-i', str(path), '-filter_complex',
                   '[0:v]setpts=0.5*PTS,fps=6,scale=384:384:flags=lanczos,split[a][b];'
                   '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                   '-loop', '0', str(gif)]
        subprocess.run(command, check=True)
        records.append(dict(stage=stage, success=verification['success'], controls=verification['controls'],
                            input_video_sha256=verification['video_sha256'], preview_playback_speed=2,
                            gif=str(gif), gif_bytes=gif.stat().st_size,
                            gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
                            final_frame_sheet=str(sheet_path)))
    report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), previews=records)
    (previews / ('report-' + '-'.join(args.stages) + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
