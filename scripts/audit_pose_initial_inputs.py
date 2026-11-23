"""Compare real reset observations with stored LeRobot training inputs, without a model."""
import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client.image_tools import resize_with_pad
from VLABench.utils.utils import quaternion_to_euler

from pi05_runtime import file_sha256
from reference_pose_lerobot import load_pose_split
from select_fruit import camera_names, digest_arrays, open_episode, physics_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, action='append', required=True)
    parser.add_argument('--case', action='append', required=True, metavar='TASK:SEED')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    episodes = [ep for root in args.dataset for ep in load_pose_split(root, 'train')]
    results = []
    for case in args.case:
        task, seed = case.rsplit(':', 1)
        selected = [ep for ep in episodes if (ep.task, ep.seed) == (task, int(seed))]
        if len(selected) != 1 or selected[0].policy_prefix_steps:
            parser.error('Select one complete expert demonstration per case')
        ep, = selected
        row = ep.row(0)
        row.pop('actions')
        with open_episode(ep.seed, ep.task, render_samples=ep.render_samples) as env:
            initial = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
            if initial != ep.initial_state or env.task.get_instruction() != ep.prompt:
                raise ValueError(f'Initialization or language differs for {case}')
            ee = env.robot.get_ee_state(env.physics)
            state = np.r_[ee[:3] - env.get_robot_frame_position(), quaternion_to_euler(ee[3:7]), ee[-1]]
            np.testing.assert_allclose(state, row['observation/state'], atol=2e-7, rtol=0)
            camera_checks = {}
            for camera, key in zip(('forward', 'right', 'franka/Franka_wrist_cam'),
                                   ('observation/image', 'observation/second_image', 'observation/wrist_image')):
                actual = resize_with_pad(env.render(camera_id=camera_names(env).index(camera),
                                                   **env.render_options), 224, 224)
                stored = row[key]
                np.testing.assert_array_equal(actual, stored)
                camera_checks[camera] = dict(max_pixel_difference=0,
                                            rgb_sha256=digest_arrays(actual))
            results.append(dict(task=ep.task, seed=ep.seed, initialization_sha256=initial,
                                state_max_abs_difference=float(np.abs(state-row['observation/state']).max()),
                                cameras=camera_checks, instruction_matches=True,
                                input_fields=sorted(row)))
    report = dict(kind='fresh_reset_to_stored_lerobot_input_alignment_not_policy_evaluation',
                  script_sha256=file_sha256(__file__), policy_updates=0, model_loaded=False,
                  measurements=results, all_passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
