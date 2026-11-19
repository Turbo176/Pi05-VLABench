"""Capture official select_fruit controls and re-execute them in fresh MuJoCo."""
import argparse
import json
from pathlib import Path
import random
import time

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
import open3d
import VLABench.robots  # register official robot classes
import VLABench.tasks  # register official task classes
from VLABench.envs import load_env

TASK = 'select_fruit'
ROBOT = 'franka'
IMAGE_SIZE = 256
HD_IMAGE_SIZE = 768


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def open_episode(seed):
    random.seed(seed)
    np.random.seed(seed)
    open3d.utility.random.seed(seed)
    env = load_env(TASK, robot=ROBOT, use_llm=False)
    # load_env performs its own reset. Seed Composer and reset for the recorded rollout.
    env.random_state.seed(seed)
    env.reset()
    return env


def robot_state(env):
    q = np.asarray(env.robot.get_qpos(env.physics)).reshape(-1)
    g = [float(env.physics.bind(env.robot.mjcf_model.find('joint', f'finger_joint{i}')).qpos[0])
         for i in (1, 2)]
    return np.concatenate([q, g])


def physics_state(env):
    return np.concatenate([env.physics.data.qpos, env.physics.data.qvel]).copy()


def success(env):
    return bool(env.task.should_terminate_episode(env.physics))


def render_views(env, image_size=IMAGE_SIZE):
    return np.stack([env.render(camera_id=i, height=image_size, width=image_size)
                     for i in range(env.physics.model.ncam)])


class ExpertCapture:
    def __init__(self, env, h5, budget):
        self.env, self.h5, self.budget = env, h5, budget
        self.count = 0
        self.ended = False
        views = (env.physics.model.ncam, IMAGE_SIZE, IMAGE_SIZE, 3)
        fields = {
            'rgb': (views, 'u1'), 'state': ((9,), 'f8'), 'action': ((9,), 'f8'),
            'time': ((), 'f8'), 'next_time': ((), 'f8'), 'post_state': (physics_state(env).shape, 'f8'),
            'terminal': ((), '?'), 'success': ((), '?'), 'discount': ((), 'f8'), 'reward': ((), 'f8'),
        }
        self.data = {name: h5.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape),
                    dtype=dtype, chunks=(1, *shape), compression='lzf' if name == 'rgb' else None)
                     for name, (shape, dtype) in fields.items()}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        if self.count >= self.budget or self.ended:
            raise RuntimeError('Expert attempted to exceed the action budget or step after termination')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError(f'Invalid actuator command: {action}')
        payload = {'rgb': render_views(self.env), 'state': robot_state(self.env),
                   'action': action.copy(), 'time': self.env.physics.time()}
        ts = self.env.step(action)
        self.ended = bool(ts.last())
        payload.update(next_time=self.env.physics.time(), post_state=physics_state(self.env),
                       terminal=self.ended, success=success(self.env), reward=float(ts.reward),
                       discount=float(ts.discount))
        for name, data in self.data.items():
            data.resize(self.count + 1, axis=0)
            data[self.count] = payload[name]
        self.count += 1
        return ts


def video(path, frames, fps, label, instruction, *, image_size=IMAGE_SIZE, crf=23):
    size = image_size * 2
    scale = image_size / IMAGE_SIZE
    temp_path = path.with_suffix('.partial.mp4')
    writer = imageio_ffmpeg.write_frames(
        str(temp_path), (size, size), fps=fps, codec='libx264', macro_block_size=1,
        ffmpeg_log_level='error', output_params=['-crf', str(crf), '-movflags', '+faststart'],
    )
    writer.send(None)
    try:
        for views in frames:
            frame = np.vstack([np.hstack(views[:2]), np.hstack(views[2:4])]).copy()
            cv2.rectangle(frame, (0, 0), (size, round(44 * scale)), (0, 0, 0), -1)
            for y, line in [(17, label), (36, instruction)]:
                cv2.putText(frame, line, (round(8 * scale), round(y * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, .48 * scale, (255, 255, 255),
                            round(scale), cv2.LINE_AA)
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
    if not temp_path.is_file() or temp_path.stat().st_size == 0:
        raise RuntimeError(f'No video written to {path}')
    temp_path.replace(path)


def probe(args):
    with open_episode(args.seed) as env:
        obs = env.get_observation(require_pcd=False)
        rgb = render_views(env)
        print(json.dumps({'instruction': env.task.get_instruction(), 'scene': env.task.scene.name,
                          'observation_keys': sorted(obs), 'rgb_shape': list(rgb.shape),
                          'q_state_shape': list(np.asarray(obs['q_state']).shape),
                          'camera_names': camera_names(env), 'control_dt_s': env.control_timestep(),
                          'initial_success': success(env)}, indent=2))


def camera_names(env):
    model = env.physics.model
    return [model.id2name(i, 'camera') for i in range(model.ncam)]


def collect(args):
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    with open_episode(args.seed) as env:
        instruction = env.task.get_instruction()
        assert not success(env), 'Episode starts with the success condition satisfied'
        stages = []
        clip = args.output / 'episode.partial.h5'
        with h5py.File(clip, 'w') as h5:
            h5.create_dataset('initial_physics_state', data=physics_state(env))
            h5.create_dataset('initial_body_xpos', data=env.physics.data.xpos.copy())
            h5.create_dataset('initial_body_xquat', data=env.physics.data.xquat.copy())
            h5.attrs.update(seed=args.seed, task=TASK, robot=ROBOT, instruction=instruction,
                            control_dt_s=env.control_timestep(), camera_names=json.dumps(camera_names(env)),
                            scene=env.task.scene.name, initial_time_s=env.physics.time())
            h5.attrs['episode_config'] = json.dumps(env.save(), default=lambda v: np.asarray(v).tolist())
            wrapped = ExpertCapture(env, h5, args.budget)
            for skill in env.get_expert_skill_sequence():
                print(f'Running official skill {skill.func.__name__}', flush=True)
                result = skill(wrapped)
                stage_ok = bool(result[2]) if result is not None else False
                stage = {'name': skill.func.__name__, 'stage_success': stage_ok,
                         'steps_so_far': wrapped.count, 'success_condition': success(env)}
                stages.append(stage)
                print(json.dumps(stage), flush=True)
                if wrapped.ended or result is None:
                    break
            ok = wrapped.ended and success(env)
            if wrapped.count:
                h5.create_dataset('final_rgb', data=render_views(env), compression='lzf')
            fps = 1 / env.control_timestep()
        summary = {'source': 'official VLABench expert', 'task': TASK, 'robot': ROBOT, 'seed': args.seed,
                   'instruction': instruction, 'success': ok, 'terminal': wrapped.ended,
                   'steps': wrapped.count, 'control_hz': fps, 'scene': env.task.scene.name,
                   'camera_names': camera_names(env), 'stages': stages, 'wall_seconds': round(time.monotonic()-start, 2)}
        write_json(args.output / 'expert.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        if not ok:
            raise SystemExit('Expert did not reach the official success condition')
        clip.rename(args.output / 'episode.h5')
    with h5py.File(args.output / 'episode.h5') as h5:
        frames = (h5['rgb'][i] if i < len(h5['rgb']) else h5['final_rgb'][:]
                  for i in range(len(h5['rgb']) + 1))
        video(args.output / 'expert.mp4', frames, fps, 'Official VLABench expert', instruction)


def replay(args):
    start = time.monotonic()
    image_size = HD_IMAGE_SIZE if args.hd else IMAGE_SIZE
    basename = 'action_replay_hd' if args.hd else 'action_replay'
    with h5py.File(args.episode) as h5, open_episode(int(h5.attrs['seed'])) as env:
        np.testing.assert_allclose(physics_state(env), h5['initial_physics_state'][:], atol=1e-9, rtol=0)
        np.testing.assert_allclose(env.physics.data.xpos, h5['initial_body_xpos'][:], atol=1e-9, rtol=0)
        np.testing.assert_allclose(env.physics.data.xquat, h5['initial_body_xquat'][:], atol=1e-9, rtol=0)
        np.testing.assert_allclose(env.physics.time(), h5.attrs['initial_time_s'], atol=1e-9, rtol=0)
        assert env.task.get_instruction() == h5.attrs['instruction'] and not success(env)
        actions = h5['action'][:]
        assert len(actions) and bool(h5['terminal'][-1]) and bool(h5['success'][-1])
        assert not np.any(h5['terminal'][:-1])
        np.testing.assert_allclose(h5['next_time'][:-1], h5['time'][1:], atol=1e-9, rtol=0)
        np.testing.assert_allclose(h5['next_time'][:] - h5['time'][:], h5.attrs['control_dt_s'], atol=1e-9, rtol=0)
        max_error = 0.0

        def frames():
            nonlocal max_error
            for index, action in enumerate(actions):
                np.testing.assert_allclose(robot_state(env), h5['state'][index], atol=1e-8, rtol=0)
                yield render_views(env, image_size)
                ts = env.step(action)
                delta = np.max(np.abs(physics_state(env) - h5['post_state'][index]))
                max_error = max(max_error, float(delta))
                if delta > 1e-8 or bool(ts.last()) != bool(h5['terminal'][index]) or success(env) != bool(h5['success'][index]):
                    raise RuntimeError(f'Replay diverged at action {index}: state error {delta}')
            yield render_views(env, image_size)

        video(args.episode.parent / f'{basename}.mp4', frames(), 1/env.control_timestep(),
              'Saved actions re-executed in MuJoCo', h5.attrs['instruction'],
              image_size=image_size, crf=16 if args.hd else 23)
        summary = {'source': 'saved actuator commands re-executed in a fresh simulator', 'task': TASK,
                   'seed': int(h5.attrs['seed']), 'steps': len(actions), 'success': success(env),
                   'max_abs_physics_state_error': max_error, 'wall_seconds': round(time.monotonic()-start, 2)}
        if args.hd:
            summary['video_resolution'] = [image_size * 2, image_size * 2]
        write_json(args.episode.parent / f'{basename}.json', summary)
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('probe').add_argument('--seed', type=int, default=42)
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--seed', type=int, default=42)
    collect_parser.add_argument('--budget', type=int, default=300)
    collect_parser.add_argument('--output', type=Path, required=True)
    replay_parser = sub.add_parser('replay')
    replay_parser.add_argument('episode', type=Path)
    replay_parser.add_argument('--hd', action='store_true', help='Render 768 x 768 per camera')
    args = parser.parse_args()
    {'probe': probe, 'collect': collect, 'replay': replay}[args.mode](args)


if __name__ == '__main__':
    main()
