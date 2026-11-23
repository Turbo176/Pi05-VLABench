"""Replay a verified fruit episode and build a three-view training cache."""
import argparse
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np
from openpi_client.image_tools import resize_with_pad
from VLABench.utils.utils import matrix_to_quaternion, quaternion_to_euler

from select_fruit import REPLAY_STATE_ATOL, camera_names, digest_arrays, open_episode, physics_state, robot_state, success

CAMERAS = ('forward', 'right', 'franka/Franka_wrist_cam')


def render_episode(source, row, output):
    seed = int(source.attrs['seed'])
    commands = source['action'][:]
    if seed != row['seed'] or len(commands) != row['frames'] or commands.dtype != np.float64:
        raise ValueError('Expected the verified original float64 trajectory')
    partial = output.with_suffix('.partial.h5')
    with open_episode(seed, 'select_fruit') as env, h5py.File(partial, 'x') as cache:
        initial = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
        for key, measured in [('initial_physics_state', physics_state(env)), ('initial_body_xpos', env.physics.data.xpos),
                              ('initial_body_xquat', env.physics.data.xquat)]:
            np.testing.assert_array_equal(measured, source[key][:], err_msg=key)
        instruction = env.task.get_instruction()
        if initial != row['initial_state_sha256'] or instruction != row['instruction'] or success(env):
            raise RuntimeError('The reset differs from the verified training export')
        np.testing.assert_allclose(env.control_timestep(), 0.1, atol=1e-9, rtol=0)
        origin = np.asarray(env.get_robot_frame_position())
        np.testing.assert_allclose(origin, [0, -0.4, 0.78], atol=1e-7, rtol=0)
        indices = [camera_names(env).index(camera) for camera in CAMERAS]
        shadow = env.physics.copy(share_model=True)
        model = shadow.model
        site = model.name2id(env.robot.end_effector_site.full_identifier, 'site')
        joints = [model.name2id(env.robot.mjcf_model.find('joint', f'joint{i}').full_identifier, 'joint')
                  for i in range(1, 8)]
        arm = model.jnt_qposadr[joints]
        rgb = cache.create_dataset('rgb', (len(commands), 3, 224, 224, 3), 'u1',
                                   chunks=(1, 3, 224, 224, 3), compression='lzf')
        states = np.empty((len(commands), 7), dtype=np.float32)
        actions = np.empty_like(states)
        max_state_error = 0.0
        for i, command in enumerate(commands):
            np.testing.assert_allclose(robot_state(env), source['state'][i], atol=REPLAY_STATE_ATOL, rtol=0)
            np.testing.assert_allclose(env.physics.time(), source['time'][i], atol=1e-9, rtol=0)
            ee = env.robot.get_ee_state(env.physics)
            states[i] = np.r_[ee[:3] - origin, quaternion_to_euler(ee[3:7]), ee[-1]]
            shadow.data.qpos[arm] = command[:7]
            mujoco.mj_kinematics(model.ptr, shadow.data.ptr)
            actions[i] = np.r_[shadow.data.site_xpos[site] - origin,
                               quaternion_to_euler(matrix_to_quaternion(shadow.data.site_xmat[site])),
                               command[-1] > 0.03]
            rendered = [env.render(camera_id=index, **env.render_options) for index in indices]
            if any(view.shape != (480, 480, 3) or view.dtype != np.uint8 for view in rendered):
                raise ValueError('Expected the upstream 480x480 RGB renders')
            rgb[i] = [resize_with_pad(view, 224, 224) for view in rendered]
            step = env.step(command)
            post_state = physics_state(env)
            np.testing.assert_allclose(post_state, source['post_state'][i], atol=REPLAY_STATE_ATOL, rtol=0)
            max_state_error = max(max_state_error, float(np.max(np.abs(post_state - source['post_state'][i]))))
            ended = i == len(commands) - 1
            if bool(step.last()) != ended or success(env) != ended:
                raise RuntimeError(f'Unexpected official termination at control {i + 1}')
        for key, array in [('state', states), ('actions', actions), ('time', source['time'][:]),
                           ('terminal', source['terminal'][:]), ('success', source['success'][:])]:
            cache.create_dataset(key, data=array)
        cache.attrs.update(seed=seed, task='select_fruit', instruction=instruction, max_state_error=max_state_error,
                           source=row['source'], initial_state_sha256=initial, cameras=json.dumps(CAMERAS), fps=10)
    partial.rename(output)
    return max_state_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', required=True, type=Path)
    parser.add_argument('--controls', type=Path, help='Optional packed float64 controls for local recordings')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.verification.read_text(encoding='utf-8'))['splits']['train']
    if not split['all_frames_verified']:
        raise ValueError('Expected a verified LeRobot export')
    row, = (r for r in split['episodes_detail'] if r['task'] == 'select_fruit' and r['seed'] == args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with h5py.File(args.controls or row['source'], 'r') as file:
        replay_error = render_episode(file[str(args.seed)] if args.controls else file, row,
                                      args.output_dir / f'{args.seed}.h5')
    print(json.dumps(dict(seed=args.seed, frames=row['frames'], replay_error=replay_error,
                          seconds=time.monotonic() - start)))


if __name__ == '__main__':
    main()
