"""Record fruit expert demonstrations through the reference policy's pose controller."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryFile
from unittest.mock import patch

import h5py
import mujoco
import numpy as np
from openpi_client.image_tools import resize_with_pad
from VLABench.utils.utils import euler_to_quaternion, matrix_to_quaternion, quaternion_to_euler

from convert_lerobot import POSE_TASKS
from export_reference_fruit_views import CAMERAS
from select_fruit import ActionBudgetExceeded, ExpertCapture, camera_names, digest_arrays, open_episode, physics_state, success


class PoseExpertCapture(ExpertCapture):
    def __init__(self, env, trace, budget, render):
        super().__init__(env, trace, budget, capture_images=False)
        self.ik = env.robot.get_qpos_from_ee_pos
        self.last_request = None
        self.origin = np.asarray(env.get_robot_frame_position())
        self.shadow = env.physics.copy(share_model=True)
        model = self.shadow.model
        self.site = model.name2id(env.robot.end_effector_site.full_identifier, 'site')
        joint_names = [env.robot.mjcf_model.find('joint', f'joint{i}').full_identifier for i in range(1, 8)]
        self.arm = model.jnt_qposadr[[model.name2id(name, 'joint') for name in joint_names]]
        self.camera_indices = [camera_names(env).index(name) for name in CAMERAS]
        self.states, self.targets = [], []
        self.ik_failures = 0
        self.rgb = None
        if render:
            shape = (3, 224, 224, 3)
            self.rgb = trace.create_dataset('rgb', (0, *shape), 'u1', maxshape=(None, *shape),
                                            chunks=(1, *shape), compression='lzf')

    def record_ik(self, physics, pos, quat=None, **kwargs):
        solved, arm = self.ik(physics=physics, pos=pos, quat=quat, **kwargs)
        if quat is None or kwargs:
            raise ValueError('Expected an official Cartesian target with standard IK settings')
        self.last_request = (arm.copy(), np.asarray(pos).copy(), np.asarray(quat).copy())
        return solved, arm

    def step(self, action):
        if self.count >= self.budget:
            raise ActionBudgetExceeded('Pose expert reached the simulator control budget')
        action = np.asarray(action, dtype=np.float64)
        if self.last_request is not None and np.array_equal(action[:7], self.last_request[0]):
            _, position, quat = self.last_request
        else:
            # The expert's finger skills hold a joint target without calling IK.
            self.shadow.data.qpos[self.arm] = action[:7]
            mujoco.mj_kinematics(self.shadow.model.ptr, self.shadow.data.ptr)
            position = self.shadow.data.site_xpos[self.site]
            quat = matrix_to_quaternion(self.shadow.data.site_xmat[self.site])
        self.last_request = None
        pose = np.r_[position - self.origin, quaternion_to_euler(quat), action[-1] > 0.03].astype(np.float32)
        ee = self.env.robot.get_ee_state(self.env.physics)
        state = np.r_[ee[:3] - self.origin, quaternion_to_euler(ee[3:7]), ee[-1]]
        if self.rgb is not None:
            self.rgb.resize(self.count + 1, axis=0)
            self.rgb[self.count] = [resize_with_pad(self.env.render(camera_id=index, **self.env.render_options), 224, 224)
                                    for index in self.camera_indices]
        solved, arm = self.ik(physics=self.env.physics, pos=pose[:3] + self.origin,
                              quat=euler_to_quaternion(*pose[3:6]))
        fingers = np.repeat(0.04 if pose[-1] >= 0.1 else 0., 2)
        timestep = super().step(np.r_[arm, fingers])
        self.ik_failures += int(not solved)
        self.states.append(state.astype(np.float32))
        self.targets.append(pose)
        return timestep


def capture_skills(expert, skills):
    with patch.object(expert.env.robot, 'get_qpos_from_ee_pos', expert.record_ik):
        try:
            for skill in skills:
                result = skill(expert)
                if expert.ended or result is None:
                    break
        except ActionBudgetExceeded:
            pass
    return expert.ended and success(expert.env)


def write_pose_labels(trace, expert, task, seed, initial, instruction):
    trace.move('state', 'joint_state')
    trace.create_dataset('state', data=expert.states)
    trace.create_dataset('actions', data=expert.targets)
    trace.attrs.update(seed=seed, task=task, instruction=instruction, initial_state_sha256=initial,
                       cameras=json.dumps(CAMERAS), fps=10,
                       render_samples=int(expert.env.physics.model.vis.quality.offsamples),
                       source='official expert with reference pose controller')


def collect(seed, cache, budget, task='select_fruit', *, render_samples=None):
    name = str(seed) if task == 'select_fruit' else f'{task}-{seed}'
    if cache is None:
        temporary = TemporaryFile()
        destination = temporary
    else:
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / f'{name}.partial.h5'
    with open_episode(seed, task, render_samples=render_samples) as env, h5py.File(destination, 'x' if cache is not None else 'w') as trace:
        initial = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
        instruction = env.task.get_instruction()
        if success(env):
            raise RuntimeError('The expert reset already satisfies the goal')
        expert = PoseExpertCapture(env, trace, budget, cache is not None)
        solved = capture_skills(expert, env.get_expert_skill_sequence())
        if cache is not None and solved:
            write_pose_labels(trace, expert, task, seed, initial, instruction)
    if cache is None:
        temporary.close()
    elif solved:
        destination.rename(cache / f'{name}.h5')
    return dict(task=task, seed=seed, success=solved, steps=expert.count, ik_nonconverged_steps=expert.ik_failures,
                instruction=instruction, initial_state_sha256=initial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=POSE_TASKS, default='select_fruit')
    parser.add_argument('--seed', required=True, type=int, nargs='+')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--budget', type=int, default=300)
    parser.add_argument('--render-samples', type=int, choices=(0, 2, 4, 8))
    args = parser.parse_args()
    if not 1 <= args.budget <= 600:
        parser.error('Expected a control budget between 1 and 600')
    for seed in args.seed:
        print(json.dumps(collect(seed, args.cache, args.budget, args.task, render_samples=args.render_samples)), flush=True)


if __name__ == '__main__':
    main()
