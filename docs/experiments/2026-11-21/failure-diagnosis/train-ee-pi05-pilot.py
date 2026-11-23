"""End-effector π0.5 pilot using saved LeRobot joint demonstrations and official IK."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from rlinf.models.embodiment.openpi.modules.model import Observation
import torch
from VLABench.utils.utils import euler_to_quaternion, matrix_to_quaternion, quaternion_to_euler

from pi05_data import LeRobotTrajectories, TASKS
from pi05_runtime import DENOISE_STEPS, build_model, restore_trainable, trainable_snapshot
from select_fruit import camera_names, open_episode, robot_state, success
from vlabench_policy_io import SCENE_CAMERA, WRIST_CAMERA, simulator_inputs

HORIZON = 50
EXECUTE = 10
DIM = 7


class FrankaGeometry:
    def __init__(self, env):
        self.origin = np.array(env.get_robot_frame_position())
        self.physics = env.physics.copy(share_model=True)
        model = env.physics.model
        self.site = model.name2id(env.robot.end_effector_site.full_identifier, 'site')
        arm_joints = [env.robot.mjcf_model.find('joint', f'joint{i}') for i in range(1, 8)]
        self.arm = [model.jnt_qposadr[model.name2id(j.full_identifier, 'joint')] for j in arm_joints]

    def convert(self, joints, *, target):
        result = np.empty((len(joints), DIM), dtype=np.float32)
        for index, q in enumerate(joints):
            self.physics.data.qpos[self.arm] = q[:7]
            mujoco.mj_kinematics(self.physics.model.ptr, self.physics.data.ptr)
            position = self.physics.data.site_xpos[self.site] - self.origin
            quaternion = matrix_to_quaternion(self.physics.data.site_xmat[self.site])
            gripper = q[-1] > 0.03 if target else np.all(q[-2:] < 0.035)
            result[index] = np.r_[position, quaternion_to_euler(quaternion), gripper]
        return result


def prepare(datasets):
    states, actions = {}, {}
    for task, seed in zip(TASKS, (42, 42, 43)):
        with open_episode(seed, task) as env:
            geometry = FrankaGeometry(env)
            q = robot_state(env)
            got = geometry.convert(q[None], target=False)[0]
            ee = env.robot.get_ee_state(env.physics)
            expected = np.r_[ee[:3] - geometry.origin, quaternion_to_euler(ee[3:7]), ee[-1]]
            np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)
            for split, reader in enumerate(datasets):
                for episode in reader.episodes[task]:
                    table, label = reader._read(episode)
                    key = split, episode
                    states[key] = geometry.convert(np.array(table['observation.state'].to_pylist()), target=False)
                    actions[key] = geometry.convert(label, target=True)
    return states, actions


def choose(rng, datasets, states, actions, task, horizon):
    pool = [(i, episode) for i, data in enumerate(datasets) for episode in data.episodes[task]]
    split, episode = pool[int(rng.integers(len(pool)))]
    observed = states[split, episode]
    index = int(rng.integers(len(observed)))
    row = datasets[split].example(episode, index)
    row['state'] = observed[index]
    row['actions'] = actions[split, episode][np.minimum(index + np.arange(horizon), len(observed) - 1)].copy()
    return row, split, episode, index


def to_relative(poses, starting_pose):
    relative = poses.copy()
    relative[:, :6] -= starting_pose[:6]
    relative[:, 3:6] = (relative[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return relative


def fit_stats(states, actions):
    state_acc, action_acc = normalize.RunningStats(), normalize.RunningStats()
    for key, state in states.items():
        state_acc.update(state)
        indices = np.minimum(np.arange(len(state))[:, None] + np.arange(EXECUTE), len(state) - 1)
        pose_chunks = actions[key][indices].copy()
        pose_chunks[..., :6] -= state[:, None, :6]
        pose_chunks[..., 3:6] = (pose_chunks[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
        action_acc.update(pose_chunks.reshape(-1, DIM))
    return {'state': state_acc.get_statistics(), 'actions': action_acc.get_statistics()}


class EndEffectorIO:
    def __init__(self, stats):
        self.tokenizer = PaligemmaTokenizer(200)
        self.stats = stats

    def encode(self, row):
        def tensor(value):
            return torch.from_numpy(np.array(value, copy=True))[None].to('cuda')

        state_stats = self.stats['state']
        state = (2 * (row['state'] - state_stats.q01) / (state_stats.q99 - state_stats.q01 + 1e-6) - 1).astype(np.float32)
        tokens, mask = self.tokenizer.tokenize(row['prompt'], state)
        observation = Observation.from_dict(dict(image={k: tensor(v) for k, v in row['image'].items()},
            image_mask={k: tensor(v) for k, v in row['image_mask'].items()}, state=tensor(state),
            tokenized_prompt=tensor(tokens.astype(np.int64)), tokenized_prompt_mask=tensor(mask)))
        labels = None
        if 'actions' in row:
            labels = to_relative(row['actions'], row['state'])
            stat = self.stats['actions']
            labels = (2 * (labels - stat.q01) / (stat.q99 - stat.q01 + 1e-6) - 1).astype(np.float32)
            labels = tensor(np.pad(labels, ((0, 0), (0, 32 - DIM))))
        return observation, labels

    def decode(self, normalized, state):
        stat = self.stats['actions']
        actions = (normalized + 1) * (stat.q99 - stat.q01 + 1e-6) / 2 + stat.q01
        actions[:, :6] += state[:6]
        return actions


def rollout(model, io, task, seed):
    noise = torch.Generator(device='cuda').manual_seed(seed)
    with open_episode(seed, task) as env, torch.inference_mode():
        prompt = env.task.get_instruction()
        cameras = camera_names(env)
        ids = {name: cameras.index(name) for name in (SCENE_CAMERA, WRIST_CAMERA)}
        origin = np.asarray(env.get_robot_frame_position()).copy()
        steps, ik_errors, lift = 0, 0, 0.0
        target = env.task.entities[env.task.target_entity]
        initial_z = target.get_xpos(env.physics)[2]
        while steps < 600:
            views = {name: env.render(camera_id=index, height=256, width=256) for name, index in ids.items()}
            row = simulator_inputs(views, robot_state(env), prompt)
            ee = env.robot.get_ee_state(env.physics)
            row['state'] = np.r_[ee[:3] - origin, quaternion_to_euler(ee[3:7]), ee[-1]].astype(np.float32)
            observation, _ = io.encode(row)
            output = model.sample_actions(observation, num_steps=DENOISE_STEPS, rng=noise)
            actions = io.decode(output[0, :EXECUTE, :DIM].float().cpu().numpy(), row['state'])
            if not np.isfinite(actions).all():
                raise FloatingPointError('Nonfinite end-effector commands')
            for action in actions:
                solved, joints = env.robot.get_qpos_from_ee_pos(env.physics, action[:3] + origin,
                                                                euler_to_quaternion(*action[3:6]))
                if not solved or not np.isfinite(joints).all():
                    ik_errors += 1
                    joints = robot_state(env)[:7]
                fingers = np.full(2, 0.04 if action[-1] >= 0.1 else 0.)
                step = env.step(np.r_[joints, fingers])
                steps += 1
                lift = max(lift, float(target.get_xpos(env.physics)[2] - initial_z))
                if step.last() or success(env) or steps == 600:
                    return dict(task=task, seed=seed, success=success(env), steps=steps,
                                ik_nonconverged_steps=ik_errors, max_target_lift_m=lift)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--updates', type=int, default=1200)
    parser.add_argument('--horizon', type=int, choices=(10, 50), default=HORIZON)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.manual_seed(263)
    rng = np.random.default_rng(263)
    original = Path('.local/compact-data/train')
    extra = Path('.local/compact-data/fruit-expansion-v21-20261121')
    readers = (LeRobotTrajectories(original, Path('incoming/train-verification.json')),
               LeRobotTrajectories(extra / 'train', extra / 'verification.json', required_tasks=('select_fruit',)))
    states, actions = prepare(readers)
    stats = fit_stats(states, actions)
    (args.output / 'norm_stats.json').write_text(normalize.serialize_json(stats))
    print(json.dumps(dict(episodes=len(states), frames=sum(map(len, states.values())))), flush=True)
    model, _ = build_model(args.base, 'cuda', EXECUTE, lora_rank=16, vlm_lora_rank=8, action_horizon=args.horizon)
    io = EndEffectorIO(stats)
    weights = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(weights, lr=1e-4)
    start = time.monotonic()
    for index in range(args.updates):
        task = TASKS[index % len(TASKS)]
        row, split, episode, frame = choose(rng, readers, states, actions, task, args.horizon)
        obs, target = io.encode(row)
        optimizer.zero_grad(set_to_none=True)
        each = model.compute_loss(obs, target, train=False)
        loss = each.mean() if args.horizon == EXECUTE else (
            each[:, :EXECUTE].mean() * 0.7 + each[:, EXECUTE:].mean() * 0.3)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(weights, 1., error_if_nonfinite=True)
        optimizer.step()
        if index == 0 or (index + 1) % 25 == 0:
            print(json.dumps(dict(update=index+1, task=task, split=split, episode=episode,
                                  frame=frame, loss=float(loss.detach()), grad=float(grad))), flush=True)
    parameters = trainable_snapshot(model)
    info = dict(scope='multitask end-effector SFT pilot', updates=args.updates,
                wall_seconds=time.monotonic() - start, lora_rank=16, vlm_lora_rank=8,
                seed=263, execute_steps=EXECUTE, action_horizon=args.horizon)
    checkpoint = args.output / 'sft.pt'
    torch.save(dict(policy=parameters, optimizer=optimizer.state_dict(), metadata=info), checkpoint)
    restore_trainable(model, torch.load(checkpoint, weights_only=True, map_location='cpu')['policy'])
    print(json.dumps(info), flush=True)
    with (args.output / 'development.jsonl').open('x', encoding='utf-8') as handle:
        for task, seed in [('select_fruit', n) for n in range(1201, 1205)] + [('insert_flower', 46), ('take_out_cool_drink', 46)]:
            result = rollout(model, io, task, seed)
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
