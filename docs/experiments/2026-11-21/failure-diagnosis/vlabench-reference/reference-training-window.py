import json
from pathlib import Path

import mujoco
import numpy as np
from openpi import transforms
from openpi.training.config import get_config
from VLABench.utils.utils import quaternion_to_euler, matrix_to_quaternion

from pi05_data import LeRobotTrajectories
from select_fruit import camera_names, digest_arrays, open_episode, render_views, robot_state, success

reader = LeRobotTrajectories(Path('.local/compact-data/train'), Path('incoming/train-verification.json'), episode_filter={0})
table, labels = reader._read(0)
states = np.array(table['observation.state'].to_pylist(), dtype=np.float32)
original = np.load('.local/reference-seed42-controls.npz')
np.testing.assert_allclose(original['action'], labels, atol=1.3e-7, rtol=0)
np.testing.assert_allclose(original['state'], states, atol=1.3e-7, rtol=0)
states, labels = original['state'], original['action']
frame_index = 34
raw = None
checked_states = 0
with open_episode(42, 'select_fruit') as env:
    origin = np.array(env.get_robot_frame_position())
    np.testing.assert_allclose(origin, (0, -0.4, 0.78), atol=1e-7, rtol=0)
    instruction = env.task.get_instruction()
    names = camera_names(env)
    shadow = env.physics.copy(share_model=True)
    site_id = shadow.model.name2id(env.robot.end_effector_site.full_identifier, 'site')
    joint_ids = [shadow.model.name2id(env.robot.mjcf_model.find('joint', f'joint{i}').full_identifier, 'joint')
                 for i in range(1, 8)]
    arm = shadow.model.jnt_qposadr[joint_ids]
    maximum_state_difference = 0.0
    for i, action in enumerate(labels):
        measured = robot_state(env)
        maximum_state_difference = max(maximum_state_difference, float(np.max(abs(measured - states[i]))))
        np.testing.assert_allclose(measured, states[i], atol=2e-6, rtol=0)
        checked_states += 1
        if i == frame_index:
            ee = env.robot.get_ee_state(env.physics)
            state = np.r_[ee[:3] - origin, quaternion_to_euler(ee[3:7]), ee[-1]].astype(np.float32)
            control_poses = []
            for label in labels[i:i + 10]:
                shadow.data.qpos[arm] = label[:7]
                mujoco.mj_kinematics(shadow.model.ptr, shadow.data.ptr)
                xyz = shadow.data.site_xpos[site_id] - origin
                euler = quaternion_to_euler(matrix_to_quaternion(shadow.data.site_xmat[site_id]))
                control_poses.append(np.r_[xyz, euler, label[-1] >= 0.03])
            rgb = render_views(env, 480)
            raw = {'observation/image': rgb[2], 'observation/second_image': rgb[0],
                   'observation/wrist_image': rgb[3], 'observation/state': state, 'prompt': instruction,
                   'actions': np.asarray(control_poses, dtype=np.float32)}
        timestep = env.step(action)
        if timestep.last() and i != len(labels) - 1:
            raise RuntimeError('Unexpected early stop on the training replay')
    solved = success(env)
    assert solved
    assert raw is not None
    assert instruction == reader.example(0, frame_index)['prompt']
config = get_config('pi05_ft_vlabench_primitive')
data = config.data.create(config.assets_dirs, config.model)
provided = transforms.compose(data.data_transforms.inputs)(raw.copy())
resized = transforms.compose(data.model_transforms.inputs)(provided.copy())
decoded = transforms.compose(data.data_transforms.outputs)({'actions': provided['actions'].copy(), 'state': provided['state']})
round_trip = float(np.max(np.abs(decoded['actions'] - raw['actions'])))
np.testing.assert_allclose(decoded['actions'], raw['actions'], atol=3e-7, rtol=0)
shapes = {k: list(v.shape) for k, v in resized['image'].items()}
assert shapes == {k: [224,224,3] for k in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')}
assert all(resized['image_mask'].values())
assert resized['actions'].shape == (10, 32)
result = {'seed': 42, 'split': 'train', 'lerobot_episode_index': 0, 'frame_index': frame_index,
          'instruction': instruction, 'replay_success': solved, 'replay_controls': checked_states,
          'max_state_error': maximum_state_difference, 'cameras': [names[i] for i in (2, 0, 3)],
          'image_hashes': {k: digest_arrays(v) for k, v in resized['image'].items()}, 'image_shapes': shapes,
          'model_action_shape': list(resized['actions'].shape), 'delta_absolute_round_trip_max': round_trip}
Path('runs/vlabench-reference-fruit-development/training-window-preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
