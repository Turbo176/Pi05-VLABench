"""Check that a real LeRobot sample and fresh simulator map to identical inputs."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from convert_lerobot import FPS, JOINTS
from vlabench_policy_io import (
    ACTION_HORIZON, MODEL_ACTION_DIM, SCENE_CAMERA, WRIST_CAMERA,
    actuator_targets, lerobot_inputs, simulator_inputs,
)


def sample_steps(count):
    return sorted({0, count // 2, count - 1})


def read_dataset(root, output):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(f'local/pi05-vlabench-{root.name}', root=root, video_backend='pyav',
                              delta_timestamps={'action': (np.arange(ACTION_HORIZON) / FPS).tolist()})
    if dataset.fps != FPS:
        raise ValueError(f'Expected dataset at {FPS} Hz')
    samples = {}
    counts = []
    for episode in range(dataset.num_episodes):
        start = int(dataset.episode_data_index['from'][episode])
        count = int(dataset.episode_data_index['to'][episode]) - start
        counts.append(count)
        for step in sample_steps(count):
            row = dataset[start + step]
            assert int(row['episode_index']) == episode and int(row['frame_index']) == step
            assert row['task'] == dataset.meta.tasks[int(row['task_index'])]
            inputs = lerobot_inputs(row)
            assert set(inputs) == {'image', 'image_mask', 'state', 'prompt', 'actions'}
            prefix = f'{episode}/{step}/'
            for key, value in inputs.items():
                if isinstance(value, dict):
                    samples.update({prefix + key + '/' + name: item for name, item in value.items()})
                else:
                    samples[prefix + key] = value
            samples[prefix + 'timestamp'] = np.asarray(row['timestamp'])
            samples[prefix + 'action_is_pad'] = np.asarray(row['action_is_pad'])
    np.savez_compressed(output, counts=counts, **samples)
    print(f'Loaded {sum(len(sample_steps(n)) for n in counts)} policy samples from {len(counts)} episodes')


def compare_episode(episode_path, samples, episode_index):
    from select_fruit import camera_names, open_episode, physics_state, robot_state, success

    with h5py.File(episode_path) as source:
        actions = source['action'][:]
        states = source['state'][:]
        times = source['time'][:]
        task, seed = str(source.attrs['task']), int(source.attrs['seed'])
        assert len(actions) == samples['counts'][episode_index]
        steps = sample_steps(len(actions))
        # Exercise removal of unused, nonzero model output coordinates.
        padded = np.pad(actions, ((0, 0), (0, MODEL_ACTION_DIM - actions.shape[-1])), constant_values=0.25)
        targets = actuator_targets(padded)
        np.testing.assert_array_equal(targets, actions)
        with open_episode(seed, task) as env:
            assert not success(env)
            np.testing.assert_array_equal(physics_state(env), source['initial_physics_state'][:])
            np.testing.assert_array_equal(env.physics.data.xpos, source['initial_body_xpos'][:])
            np.testing.assert_array_equal(env.physics.data.xquat, source['initial_body_xquat'][:])
            np.testing.assert_allclose(env.control_timestep(), 1/FPS, atol=1e-9, rtol=0)
            instruction = env.task.get_instruction()
            assert instruction == source.attrs['instruction']
            names = camera_names(env)
            for step, target in enumerate(targets):
                if step in steps:
                    views = {name: env.render(camera_id=names.index(name), height=256, width=256)
                             for name in (SCENE_CAMERA, WRIST_CAMERA)}
                    live = simulator_inputs(views, robot_state(env), instruction)
                    assert set(live) == {'image', 'image_mask', 'state', 'prompt'}
                    prefix = f'{episode_index}/{step}/'
                    for key in ('image', 'image_mask'):
                        for name, value in live[key].items():
                            np.testing.assert_array_equal(value, samples[prefix + key + '/' + name])
                    np.testing.assert_array_equal(live['state'], samples[prefix + 'state'])
                    assert live['prompt'] == samples[prefix + 'prompt'].item()
                    np.testing.assert_allclose(env.physics.time(), times[step], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(samples[prefix + 'timestamp'], times[step] - times[0],
                                               atol=1e-5, rtol=0)
                    indices = step + np.arange(ACTION_HORIZON)
                    np.testing.assert_array_equal(samples[prefix + 'action_is_pad'], indices >= len(targets))
                    np.testing.assert_array_equal(actuator_targets(samples[prefix + 'actions']),
                                                  targets[np.minimum(indices, len(targets) - 1)].astype(np.float32))
                result = env.step(target)
                np.testing.assert_array_equal(physics_state(env), source['post_state'][step])
                assert bool(result.last()) == bool(source['terminal'][step])
                assert success(env) == bool(source['success'][step])
            assert success(env) and result.last()
    print(f'{task} seed {seed}: {len(targets)} steps, samples at {steps}, successful replay')
    return {'task': task, 'seed': seed, 'instruction': instruction, 'steps': len(targets),
            'matched_frames': steps, 'replay_success': True, 'max_abs_physics_state_error': 0.0}, states, actions


def observed_ranges(values):
    return {joint: {'min': float(low), 'max': float(high)}
            for joint, low, high in zip(JOINTS, values.min(axis=0), values.max(axis=0))}


def verify_simulator(dataset_samples, episodes, report_path):
    with np.load(dataset_samples, allow_pickle=False) as samples:
        if len(episodes) != len(samples['counts']):
            raise ValueError('Supply the original H5 episodes in LeRobot episode order')
        results = [compare_episode(path, samples, index) for index, path in enumerate(episodes)]
    reports, states, actions = zip(*results)
    report = {
        'scope': 'LeRobot v2.1 and fresh MuJoCo, before OpenPI model transforms',
        'fps': FPS,
        'cameras': {'base_0_rgb': SCENE_CAMERA, 'left_wrist_0_rgb': WRIST_CAMERA, 'right_wrist_0_rgb': None},
        'action_window': ACTION_HORIZON,
        'replayed_steps': sum(result['steps'] for result in reports),
        'matched_samples': sum(len(result['matched_frames']) for result in reports),
        'episodes': reports,
        'state_ranges_rad_or_m': observed_ranges(np.concatenate(states)),
        'absolute_target_ranges_rad_or_m': observed_ranges(np.concatenate(actions)),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Policy I/O verification passed. Report: {report_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='phase', required=True)
    reader = phases.add_parser('dataset')
    reader.add_argument('root', type=Path)
    reader.add_argument('samples', type=Path)
    simulator = phases.add_parser('simulator')
    simulator.add_argument('samples', type=Path)
    simulator.add_argument('report', type=Path)
    simulator.add_argument('episodes', type=Path, nargs='+')
    args = parser.parse_args()
    if args.phase == 'dataset':
        read_dataset(args.root, args.samples)
    else:
        verify_simulator(args.samples, args.episodes, args.report)


if __name__ == '__main__':
    main()
