"""Run the published VLABench Pi0.5 policy on our paired development resets."""

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import time

from dm_control.rl.control import PhysicsError
import numpy as np
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

from convert_lerobot import CAMERAS
from select_fruit import camera_names, digest_arrays, open_episode, physics_state, render_views, success, video


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = 'f658a8bec801aab960fe0ceab370a123b31c1837'
OPENPI_REVISION = '788ef1f20e655de5bb40869fb7b7fcdd49732387'
TASK = 'select_fruit'
EVALUATION_SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_upstream_adapter(fork):
    spec = importlib.util.spec_from_file_location('vlabench_reference_adapter', fork / 'examples/vlabench/eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Pi0


class HoldStillFixture:
    def infer(self, observation):
        keys = {'observation/image', 'observation/second_image', 'observation/wrist_image',
                'observation/state', 'prompt'}
        if observation.keys() != keys or not np.isfinite(observation['observation/state']).all():
            raise RuntimeError('Unexpected reference policy observation')
        return {'actions': np.repeat(observation['observation/state'][None, :], 10, axis=0)}


def build_episode(env, agent, seed, step_limit, record, *, task=TASK, prompt_mode='official', trace=None,
                  record_physics_failure=False):
    frame_origin = env.get_robot_frame_position()
    instruction = env.task.get_instruction()
    names = camera_names(env)
    if tuple(names) != CAMERAS:
        raise RuntimeError('The upstream adapter expects right, left, forward, and wrist cameras in that order')
    if abs(env.control_timestep() - 0.1) > 1e-8 or success(env):
        raise RuntimeError('Expected an unsolved 10 Hz evaluation reset')
    first_frame = render_views(env)

    def render_model_views():
        return np.stack([env.render(camera_id=index, **env.render_options) for index in range(len(names))])

    first_model_frame = render_model_views()
    hashes = {'initial_policy_rgb_sha256': digest_arrays(first_model_frame[2], first_model_frame[0], first_model_frame[3]),
              'initial_state_sha256': digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat),
              'initial_rgb_sha256': digest_arrays(first_frame)}
    result = {'seed': seed, 'task': task, 'instruction': instruction, 'prompt_mode': prompt_mode,
              'camera_names': names, 'model_render_options': env.render_options,
              'evaluation_script_sha256': EVALUATION_SCRIPT_SHA256,
              'render_samples': int(env.physics.model.vis.quality.offsamples), **hashes}
    target_name = env.task.target_entity
    if isinstance(target_name, list):
        target_name, = target_name
    target = env.task.entities[target_name]
    initial_target = target.get_xpos(env.physics).copy()
    container_name = getattr(env.task, 'target_container', None)
    container = env.task.entities.get(container_name) if isinstance(container_name, str) else None
    fridge = (next((entity for name, entity in env.task.entities.items() if 'fridge' in name), None)
              if task == 'take_out_cool_drink' else None)

    def run():
        max_displacement = max_lift = 0.0
        min_gripper_distance = float('inf')
        min_closed_gripper_distance = None
        gripper_target_contact_steps = 0
        policy_calls = 0
        ik_nonconverged_steps = 0
        action_digest = hashlib.sha256()
        last_action = None
        action_rgb = None
        physics_error = None
        timestep = None
        solved = False
        fridge_progress = None
        if fridge is not None:
            fridge_progress = dict(door_max_abs_angle_rad=abs(float(env.physics.bind(fridge.door_joint).qpos[0])),
                                   door_opened=False, drink_was_outside=False,
                                   door_closed_at_end=bool(fridge.is_closed(env.physics)),
                                   drink_outside_at_end=not bool(fridge.contain(initial_target, env.physics)))
        if record:
            yield first_frame
        for step in range(1, step_limit + 1):
            ee_state = env.robot.get_ee_state(env.physics)
            measured_ee = ee_state.copy() if trace is not None else None
            replanned = not agent.action_plan
            if replanned:
                action_rgb = first_model_frame if step == 1 else render_model_views()
                policy_calls += 1
            if last_action is None:
                last_action = np.concatenate([ee_state[:3], quaternion_to_euler(ee_state[3:7])])
            inputs = {'rgb': action_rgb, 'ee_state': ee_state, 'robot_frame': frame_origin,
                      'instruction': instruction, 'last_action': last_action}
            pos, euler, fingers = agent.predict(inputs)
            last_action = np.concatenate([pos, euler])
            ik_success, arm = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=euler_to_quaternion(*euler))
            ik_nonconverged_steps += int(not ik_success)
            command = np.concatenate([np.asarray(arm).reshape(-1), fingers])
            if command.shape != (9,) or not np.isfinite(command).all():
                raise RuntimeError('Upstream IK produced an invalid actuator command')
            if step == 1:
                result['first_action_sha256'] = hashlib.sha256(command.tobytes()).hexdigest()
            action_digest.update(command.tobytes())
            try:
                timestep = env.step(command)
            except PhysicsError as error:
                if not record_physics_failure:
                    raise
                physics_error = str(error)
                logging.warning('%s seed %d stopped at control %d: %s', task, seed, step, error)
                if trace is not None:
                    trace.write(json.dumps(dict(step=step, physics_error=physics_error, success=False)) + '\n')
                break
            target_position = target.get_xpos(env.physics)
            if fridge_progress is not None:
                door_angle = abs(float(env.physics.bind(fridge.door_joint).qpos[0]))
                outside = not bool(fridge.contain(target_position, env.physics))
                fridge_progress.update(door_max_abs_angle_rad=max(fridge_progress['door_max_abs_angle_rad'], door_angle),
                                       door_opened=fridge_progress['door_opened'] or bool(fridge.is_open(env.physics)),
                                       drink_was_outside=fridge_progress['drink_was_outside'] or outside,
                                       door_closed_at_end=bool(fridge.is_closed(env.physics)),
                                       drink_outside_at_end=outside)
            gripper_distance = float(np.linalg.norm(env.robot.get_end_effector_pos(env.physics) - target_position))
            min_gripper_distance = min(min_gripper_distance, gripper_distance)
            if fingers[-1] < 0.03:
                min_closed_gripper_distance = (gripper_distance if min_closed_gripper_distance is None
                                               else min(min_closed_gripper_distance, gripper_distance))
            grasped = bool(target.is_grasped(env.physics, env.robot))
            gripper_target_contact_steps += int(grasped)
            displacement = target_position - initial_target
            max_displacement = max(max_displacement, float(np.linalg.norm(displacement)))
            max_lift = max(max_lift, float(displacement[2]))
            solved = success(env)
            if trace is not None:
                entry = dict(step=step, replanned=replanned, ik_converged=bool(ik_success),
                             target_xyz=target_position.tolist(), measured_ee_xyz=measured_ee[:3].tolist(),
                             measured_ee_quaternion=measured_ee[3:7].tolist(),
                             measured_ee_gripper=float(measured_ee[-1]),
                             commanded_ee_xyz=pos.tolist(), commanded_euler=euler.tolist(),
                             closed_command=bool(fingers[-1] < 0.03), grasped_target=grasped,
                             post_step_gripper_distance_m=gripper_distance, success=solved)
                if container is not None:
                    entry['container_xyz'] = container.get_xpos(env.physics).tolist()
                trace.write(json.dumps(entry) + '\n')
            if record:
                yield render_views(env)
            if step == 1 or step % 100 == 0:
                logging.info('Seed %d at control step %d/%d', seed, step, step_limit)
            if solved or timestep.last():
                break
        environment_last = bool(timestep is not None and timestep.last())
        stop = 'success' if solved else 'environment_termination' if environment_last else 'step_budget'
        failure = None if solved else ('target_displacement_under_3cm' if max_displacement < 0.03
                                      else 'target_moved_goal_unsatisfied')
        if physics_error is not None:
            stop = failure = 'simulator_instability'
            result['physics_error'] = physics_error
        result.update(success=solved, stop_reason=stop, failure_type=failure, steps=step,
                      policy_calls=policy_calls, environment_last=environment_last,
                      ik_nonconverged_steps=ik_nonconverged_steps,
                      target_max_displacement_m=max_displacement, target_max_lift_m=max_lift,
                      min_gripper_to_target_center_m=(min_gripper_distance if np.isfinite(min_gripper_distance)
                                                      else None),
                      min_close_command_to_target_center_m=min_closed_gripper_distance,
                      gripper_target_contact_steps=gripper_target_contact_steps,
                      action_sha256=action_digest.hexdigest())
        if fridge_progress is not None:
            result['fridge_subgoals'] = fridge_progress

    return result, run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork', type=Path, required=True, help='Extracted OpenPI source at the published revision')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--fixture', action='store_true', help='Exercise I/O and IK with a stationary seven-dimensional action')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(1201, 1205), action='append')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--paired-episodes', type=Path, help='Expert JSON record for checking each reset before stepping')
    args = parser.parse_args()
    if args.fixture == bool(args.checkpoint):
        parser.error('Supply --checkpoint or --fixture')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    adapter = load_upstream_adapter(args.fork)
    if args.fixture:
        client = HoldStillFixture()
        device = None
    else:
        import jax
        import openpi
        from openpi.policies import policy_config
        from openpi.training.config import get_config

        if args.fork.resolve() not in Path(openpi.__file__).resolve().parents:
            raise RuntimeError('Prepend the published OpenPI fork to PYTHONPATH')
        manifest = json.loads((args.checkpoint / 'download-verified.json').read_text())
        if manifest['revision'] != MODEL_REVISION:
            raise RuntimeError('Expected the verified published checkpoint revision')
        device = str(jax.devices()[0])
        logging.info('Loading reference on %s', device)
        client = policy_config.create_trained_policy(get_config('pi05_ft_vlabench_primitive'), args.checkpoint)
        logging.info('Loaded reference model')
    seeds = args.seed or range(1201, 1205)
    max_steps = 6 if args.fixture else json.loads((ROOT / 'configs/evaluation-v1.json').read_text())['max_control_steps']
    expected = {row['seed']: row for row in json.loads(args.paired_episodes.read_text())} if args.paired_episodes else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = 'stationary I/O fixture' if args.fixture else 'VLABench published Pi0.5 reference'
    with (args.output_dir / 'episodes.jsonl').open('x') as stream:
        for seed in seeds:
            start = time.monotonic()
            if not args.fixture:
                client._rng = jax.random.key(seed)
            with open_episode(seed, TASK) as env:
                agent = adapter(client, replan_steps=5)
                result, rollout = build_episode(env, agent, seed, max_steps, args.video)
                if expected is not None:
                    for key in ('initial_state_sha256', 'instruction'):
                        if result[key] != expected[seed][key]:
                            raise RuntimeError(f'Seed {seed} does not match the expert reset for {key}')
                if args.video:
                    video(args.output_dir / f'{TASK}-{seed}.mp4', rollout, 10,
                          f'{source} | seed {seed}', result['instruction'])
                else:
                    for _ in rollout:
                        pass
            result.update(source=source, model_revision=None if args.fixture else MODEL_REVISION,
                          openpi_revision=OPENPI_REVISION, device=device, max_control_steps=max_steps,
                          control_hz=10, replan_steps=5, wall_seconds=time.monotonic() - start)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            logging.info('Seed %d result: %s after %d control steps', seed, result['stop_reason'], result['steps'])


if __name__ == '__main__':
    main()
