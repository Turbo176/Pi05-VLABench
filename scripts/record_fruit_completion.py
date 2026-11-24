"""Record a preselected fruit rollout and verify it against its scored episode."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

from train_reference_aligned_fruit import evaluate, load_aligned_policy, paired_episodes
import evaluate_vlabench_reference as rollout
import select_fruit
import train_reference_aligned_fruit as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'base', 'fork', 'run'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=('base', 'sft', 'ppo'), required=True)
    parser.add_argument('--ppo-weights', type=Path,
                        help='Use a selected PPO checkpoint; defaults to root/ppo/ppo-1.pt')
    parser.add_argument('--label', help='Explicit video provenance, such as shared-sft180 or fruit-ppo-v2-iter2')
    parser.add_argument('--scored-episodes', type=Path,
                        help='Completed raw case record when the larger fixed suite is still running')
    parser.add_argument('--display-image-size', type=int, choices=(256, 384, 512, 768), default=512,
                        help='Display camera resolution only; policy cameras remain at the scored resolution')
    args = parser.parse_args()
    if args.ppo_weights and args.stage != 'ppo':
        parser.error('--ppo-weights requires --stage ppo')
    root = args.root
    plan = json.loads((root / 'video-plan.json').read_text())
    seed = plan['seed']
    policy = load_aligned_policy(args.run, args.base, args.fork, args.stage,
                                 (args.ppo_weights or root / 'ppo/ppo-1.pt') if args.stage == 'ppo' else None)
    cases = paired_episodes(root / 'cases/select_fruit-official.jsonl')
    scored_source = args.scored_episodes or root / ('ordinary-' + args.stage) / 'episodes.jsonl'
    expected = paired_episodes(scored_source)[seed]
    output = root / ('video-' + args.stage)
    labels = dict(base='published-reference-already-finetuned', sft='multitask-sft90', ppo='fruit-ppo1')
    if args.label:
        labels[args.stage] = args.label
    original_views, original_video = rollout.render_views, trainer.video
    rollout.render_views = partial(select_fruit.render_views, image_size=args.display_image_size)
    trainer.video = partial(select_fruit.video, image_size=args.display_image_size, crf=18)
    try:
        evaluate(policy, output, cases, 600, True, labels[args.stage], seeds=[seed], trace=True,
                 render_samples=0)
    finally:
        rollout.render_views, trainer.video = original_views, original_video
    actual = paired_episodes(output / 'episodes.jsonl')[seed]
    fields = ('success', 'steps', 'instruction', 'action_sha256', 'first_action_sha256',
              'ik_nonconverged_steps', 'initial_state_sha256', 'initial_policy_rgb_sha256',
              'render_samples', 'norm_stats_sha256', 'base_sha256', 'sampler')
    if args.stage != 'base':
        fields += ('sft_sha256',)
    if args.stage == 'ppo':
        fields += ('ppo_sha256',)
    for field in fields:
        if expected[field] != actual[field]:
            raise ValueError('Video differs from scored rollout in ' + field)
    video_path = output / ('select_fruit-' + str(seed) + '.mp4')
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = []
    for fraction in (0., .33, .66, .99):
        frame = min(frame_count - 1, int(frame_count * fraction))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            raise ValueError('Unable to decode a recorded video frame')
        samples.append(image)
    capture.release()
    if (abs(fps - 10) > .01 or frame_count < actual['steps']
            or (width, height) != (2 * args.display_image_size, 2 * args.display_image_size)):
        raise ValueError('Unexpected video frame count or control frequency')
    sheet = cv2.vconcat([cv2.hconcat(samples[:2]), cv2.hconcat(samples[2:])])
    cv2.imwrite(str(output / 'observed-frames.jpg'), sheet)
    report = dict(stage=args.stage, label=labels[args.stage], seed=seed,
                  success=actual['success'], controls=actual['steps'],
                  matched_quantitative_fields=list(fields),
                  scored_source=str(scored_source),
                  scored_source_sha256=hashlib.sha256(scored_source.read_bytes()).hexdigest(),
                  video_sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),
                  video_bytes=video_path.stat().st_size, frames=frame_count, fps=fps,
                  width=width, height=height,
                  display_only_image_size=args.display_image_size,
                  policy_camera_resolution_unchanged=True,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
