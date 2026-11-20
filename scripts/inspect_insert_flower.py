"""Locate insert_flower failures using saved simulator states and the official condition."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from select_fruit import open_episode, physics_state, success, write_json


def inspect(directory):
    record = json.loads((directory / 'expert.json').read_text(encoding='utf-8'))
    if record['task'] != 'insert_flower':
        raise ValueError(f'Expected an insert_flower run: {directory}')
    trace, = directory.glob('*.h5')
    with h5py.File(trace) as source, open_episode(record['seed'], record['task']) as env:
        physics = env.physics
        np.testing.assert_array_equal(physics_state(env), source['initial_physics_state'][:])
        np.testing.assert_array_equal(physics.data.xpos, source['initial_body_xpos'][:])
        np.testing.assert_array_equal(physics.data.xquat, source['initial_body_xquat'][:])
        flower = env.task.entities[env.task.target_entity]
        vase = env.task.entities[env.task.target_container]
        nq = physics.model.nq
        samples = []
        stages = [{'name': 'reset', 'steps_so_far': 0, 'success_condition': False}, *record['stages']]
        for stage in stages:
            step = stage['steps_so_far']
            if step:
                position_velocity = source['post_state'][step - 1]
                physics.data.qpos[:] = position_velocity[:nq]
                physics.data.qvel[:] = position_velocity[nq:]
                physics.forward()
            flower_xyz = np.asarray(flower.get_xpos(physics)).reshape(3)
            bounds = np.asarray([physics.bind(site).xpos for site in vase.key_sites(physics)])
            outside = np.maximum(np.maximum(bounds.min(axis=0) - flower_xyz,
                                            flower_xyz - bounds.max(axis=0)), 0)
            reached = success(env)
            if reached != stage['success_condition']:
                raise RuntimeError(f'Official condition changed when inspecting {directory} step {step}')
            ee_xyz = np.asarray(env.robot.get_end_effector_pos(physics)).reshape(3)
            samples.append({'stage': stage['name'], 'step': step, 'flower_xyz_m': flower_xyz.tolist(),
                            'outside_vase_xyz_m': outside.tolist(), 'official_success': reached,
                            'finger_contact_with_flower': bool(flower.is_grasped(physics, env.robot)),
                            'ee_to_flower_m': float(np.linalg.norm(ee_xyz - flower_xyz))})
    final = samples[-1]
    print(f"seed {record['seed']} {'success' if final['official_success'] else 'failure'}, "
          f"final outside-vase distance by axis (cm): {np.round(np.array(final['outside_vase_xyz_m']) * 100, 2)}",
          flush=True)
    return {'seed': record['seed'], 'instruction': record['instruction'], 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    write_json(args.output, {'source': 'official expert saved MuJoCo states at skill boundaries',
                             'units': 'meters', 'episodes': [inspect(run) for run in args.runs]})


if __name__ == '__main__':
    main()
