"""Collect expert corrections from new training starts after a π0.5 roll-in."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from dm_control.rl.control import PhysicsError

from check_reference_fruit_label_replay import run as replay_labels
from collect_reference_fruit_pose_expert import PoseExpertCapture, capture_skills, write_pose_labels
from evaluate_vlabench_reference import TASK, build_episode, load_upstream_adapter
from select_fruit import digest_arrays, open_episode, physics_state, success
from train_reference_aligned_fruit import EXECUTE, load_aligned_policy


class PolicyPrefix:
    def __init__(self, env):
        self.env = env
        self.controls = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, control):
        step = self.env.step(control)
        self.controls.append(np.asarray(control).copy())
        return step


def collect(policy, adapter, output, seed, lead, task=TASK):
    policy.reset_rng(seed)
    partial = output / f'{seed}.partial.h5'
    with open_episode(seed, task, render_samples=policy.training_render_samples) as env:
        prefix = PolicyPrefix(env)
        start, rollout = build_episode(prefix, adapter(policy, replan_steps=EXECUTE), seed, lead, False,
                                       task=task)
        for _ in rollout:
            pass
        if start['environment_last'] or success(env):
            return dict(seed=seed, policy_steps=start['steps'], expert_success=False, reason=start['stop_reason'])
        start_hash = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
        target = env.task.entities[env.task.target_entity]
        held = bool(target.is_grasped(env.physics, env.robot))
        skills = env.get_expert_skill_sequence()
        if held:
            skills = skills[1:]
        with h5py.File(partial, 'x') as trace:
            expert = PoseExpertCapture(env, trace, 600 - start['steps'], render=True)
            solved = capture_skills(expert, skills)
            if solved:
                write_pose_labels(trace, expert, task, seed, start['initial_state_sha256'], start['instruction'])
                trace.create_dataset('policy_prefix', data=np.array(prefix.controls))
                trace.attrs['prefix_end_state_sha256'] = start_hash
                trace.attrs['source'] = 'official expert correction after project SFT policy roll-in'
                trace.attrs['policy_parent_sha256'] = policy.provenance['sft_sha256']
                trace.attrs['norm_stats_sha256'] = policy.provenance['norm_stats_sha256']
                trace.attrs['base_sha256'] = policy.provenance['base_sha256']
    result = dict(seed=seed, policy_steps=start['steps'], expert_steps=expert.count, held_at_takeover=held,
                  expert_success=solved)
    if solved:
        replay = replay_labels(partial)
        result['replay_success'] = replay['success']
        if replay['success']:
            partial.rename(output / f'{seed}.h5')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('base', 'fork', 'run', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--task', choices=(TASK, 'insert_flower'), default=TASK)
    parser.add_argument('--seed', type=int, nargs='+', required=True)
    parser.add_argument('--lead', type=int, nargs='+', default=[40, 80, 120])
    args = parser.parse_args()
    if len(set(args.seed)) != len(args.seed) or not all(0 < lead <= 200 for lead in args.lead):
        parser.error('Use distinct training starts and a prefix budget between 1 and 200 controls')
    allowed = range(2080, 2120) if args.task == TASK else range(211, 241)
    if any(seed not in allowed for seed in args.seed):
        parser.error('Use fruit training seeds 2080-2119 or new flower training seeds 211-240')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = load_aligned_policy(args.run, args.base, args.fork)
    adapter = load_upstream_adapter(args.fork)
    results = []
    with (args.output / 'episodes.jsonl').open('x', encoding='utf-8') as stream:
        for index, seed in enumerate(args.seed):
            try:
                result = collect(policy, adapter, args.output, seed, args.lead[index % len(args.lead)],
                                 task=args.task)
            except PhysicsError as error:
                result = dict(seed=seed, expert_success=False, replay_success=False,
                              reason='simulator_instability_during_collection', physics_error=str(error))
            result.update(task=args.task, render_samples=policy.training_render_samples,
                          policy_parent_sha256=policy.provenance['sft_sha256'])
            results.append(result)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            print(json.dumps(result), flush=True)
    summary = dict(kind='expert_corrections_after_training_only_policy_rollin',
                   task=args.task,
                   attempts=len(results), replay_verified=sum(bool(row.get('replay_success')) for row in results),
                   policy_parent_sha256=policy.provenance['sft_sha256'],
                   render_samples=policy.training_render_samples,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
