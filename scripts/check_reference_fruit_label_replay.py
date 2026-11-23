"""Check whether saved expert pose labels still solve the task through official IK."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from VLABench.utils.utils import euler_to_quaternion

from select_fruit import digest_arrays, open_episode, physics_state, success


def run(path):
    with h5py.File(path, 'r') as trace:
        labels = trace['actions'][:]
        seed = int(trace.attrs['seed'])
        task = str(trace.attrs.get('task', 'select_fruit'))
        expected = str(trace.attrs['initial_state_sha256'])
        prefix = trace['policy_prefix'][:] if 'policy_prefix' in trace else None
        prefix_end = trace.attrs.get('prefix_end_state_sha256')
        render_samples = int(trace.attrs['render_samples']) if 'render_samples' in trace.attrs else None
    with open_episode(seed, task, render_samples=render_samples) as env:
        initial = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
        if initial != expected:
            raise ValueError(f'Unexpected reset for seed {seed}')
        if prefix is not None:
            for command in prefix:
                if env.step(command).last() or success(env):
                    raise ValueError(f'Unexpected termination during policy prefix for seed {seed}')
            actual = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
            if actual != prefix_end:
                raise ValueError(f'Policy prefix did not replay to the expert start for seed {seed}')
        origin = env.get_robot_frame_position()
        ik_failures = 0
        for index, label in enumerate(labels):
            ok, arm = env.robot.get_qpos_from_ee_pos(env.physics, label[:3] + origin,
                                                     euler_to_quaternion(*label[3:6]))
            ik_failures += int(not ok)
            control = np.r_[np.asarray(arm).reshape(-1), np.repeat(0.04 if label[6] >= 0.1 else 0, 2)]
            step = env.step(control)
            if step.last() or success(env):
                break
        return dict(task=task, seed=seed, success=success(env), steps=index + 1, available_controls=len(labels),
                    ik_nonconverged_steps=ik_failures, policy_prefix_steps=0 if prefix is None else len(prefix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='+', type=Path)
    args = parser.parse_args()
    for episode in args.episodes:
        print(json.dumps(run(episode)), flush=True)


if __name__ == '__main__':
    main()
