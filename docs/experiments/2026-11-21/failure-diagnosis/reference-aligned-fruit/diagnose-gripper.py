import json
import h5py
import numpy as np
from VLABench.utils.utils import euler_to_quaternion
from select_fruit import open_episode, success

with h5py.File('incoming/reference-aligned-original-controls.h5') as originals:
    for seed in (55, 58, 72, 75):
        labels_file = f'.local/reference-aligned/cache/{seed}.h5'
        with h5py.File(labels_file) as trace:
            labels = trace['actions'][:]
        original = originals[str(seed)]['action'][:]
        for mode in ('fk_with_original_fingers', 'original_arm_with_binary_fingers'):
            with open_episode(seed, 'select_fruit') as env:
                origin = env.get_robot_frame_position()
                ik_errors = 0
                for index, (label, saved) in enumerate(zip(labels, original, strict=True)):
                    if mode == 'fk_with_original_fingers':
                        ok, arm = env.robot.get_qpos_from_ee_pos(env.physics, label[:3] + origin,
                                                                 euler_to_quaternion(*label[3:6]))
                        ik_errors += int(not ok)
                        fingers = saved[7:]
                    else:
                        arm = saved[:7]
                        fingers = np.repeat(0.04 if label[6] >= 0.1 else 0, 2)
                    step = env.step(np.r_[arm, fingers])
                    if step.last() or success(env):
                        break
                print(json.dumps(dict(seed=seed, mode=mode, success=success(env), controls=index+1,
                                      ik_nonconverged_steps=ik_errors)), flush=True)
