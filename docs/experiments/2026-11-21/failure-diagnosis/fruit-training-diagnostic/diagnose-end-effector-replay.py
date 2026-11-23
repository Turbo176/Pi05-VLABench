import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from VLABench.utils.utils import euler_to_quaternion, matrix_to_quaternion, quaternion_to_euler

from select_fruit import open_episode, success

train = Path('.local/compact-data/train')
manifest = json.loads(Path('incoming/train-verification.json').read_text())['splits']['train']['episodes_detail']
info = json.loads((train / 'meta/info.json').read_text())
for task, seed in [('select_fruit', 42), ('select_fruit', 53), ('insert_flower', 42), ('take_out_cool_drink', 43)]:
    i, entry = next((i, entry) for i, entry in enumerate(manifest) if entry['task'] == task and entry['seed'] == seed)
    table = train / info['data_path'].format(episode_chunk=i // info['chunks_size'], episode_index=i)
    controls = np.asarray(pq.ParquetFile(table).read(columns=['action'])['action'].to_pylist())
    with open_episode(seed, task) as env:
        assert entry['instruction'] == env.task.get_instruction()
        robot = env.robot
        physics = env.physics
        frame_origin = np.asarray(env.get_robot_frame_position()).copy()
        site_id = physics.model.name2id(robot.end_effector_site.full_identifier, 'site')
        joints = [physics.model.name2id(robot.mjcf_model.find('joint', f'joint{k}').full_identifier, 'joint')
                  for k in range(1, 8)]
        joint_addrs = physics.model.jnt_qposadr[joints]
        fk = physics.copy(share_model=True)
        ee_targets = []
        for action in controls:
            fk.data.qpos[joint_addrs] = action[:7]
            fk.forward()
            position = fk.data.site_xpos[site_id] - frame_origin
            quat = matrix_to_quaternion(fk.data.site_xmat[site_id])
            ee_targets.append(np.r_[position, quaternion_to_euler(quat), float(action[-1] > 0.03)])
        misses, error = 0, []
        for step, target in enumerate(ee_targets, 1):
            quat = euler_to_quaternion(*target[3:6])
            solved, joint_targets = robot.get_qpos_from_ee_pos(physics, target[:3] + frame_origin, quat)
            misses += not solved
            error.append(float(abs(joint_targets - controls[step-1, :7]).mean()))
            fingers = np.full(2, 0.04 if target[-1] >= 0.1 else 0.0)
            if env.step(np.r_[joint_targets, fingers]).last():
                break
        print(json.dumps(dict(task=task, seed=seed, official_success=success(env), steps=step,
                              max_steps=len(controls), ik_nonconverged_steps=misses,
                              joint_target_mae_rad=float(np.mean(error)))), flush=True)
