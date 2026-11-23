"""Repeat a pinned pose-policy rollout and locate its first differing input or action."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from dm_control.rl.control import PhysicsError
import numpy as np

from convert_lerobot import POSE_TASKS
from evaluate_vlabench_reference import build_episode, load_upstream_adapter
from select_fruit import digest_arrays, open_episode, set_fruit_prompt
from summarize_pi05 import initial_rgb_key
from train_reference_aligned_fruit import IMAGE_KEYS, EXECUTE, load_aligned_policy, paired_episodes


class InputActionTrace:
    def __init__(self, policy):
        self.policy = policy
        self.calls = []
        self.actions = []
        self.states = []

    def infer(self, observation):
        state = np.array(observation['observation/state'], copy=True)
        cameras = {key: digest_arrays(observation[key]) for key in IMAGE_KEYS}
        prediction = self.policy.infer(observation)
        action = np.array(prediction['actions'], copy=True)
        self.calls.append(dict(call=len(self.calls), state_sha256=digest_arrays(state),
                               image_sha256=cameras, prompt=observation['prompt'],
                               actions_sha256=digest_arrays(action)))
        self.actions.append(action)
        self.states.append(state)
        return prediction


def integration_signature(env):
    # qpos/qvel alone omit solver warm starts and external forces.
    data = env.physics.data
    fields = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart', 'qacc', 'qfrc_applied',
              'xfrc_applied', 'mocap_pos', 'mocap_quat', 'eq_active')
    return dict(time=float(data.time), **{key: digest_arrays(np.asarray(getattr(data, key)))
                                         for key in fields if hasattr(data, key)})


def compare(first, other):
    keys = ('initial_state_sha256', 'initial_policy_rgb_sha256', 'first_action_sha256',
            'action_sha256', 'success', 'steps', 'stop_reason', 'integration_signature')
    report = {'equal_result_fields': all(first.get(key) == other.get(key) for key in keys),
              'changed_result_fields': [key for key in keys if first.get(key) != other.get(key)],
              'first_differing_policy_call': None}
    for index, (left, right) in enumerate(zip(first['calls'], other['calls'])):
        changed = [key for key in left if left[key] != right[key]]
        if changed:
            report['first_differing_policy_call'] = dict(call=index, fields=changed)
            break
    report['policy_call_counts'] = [len(first['calls']), len(other['calls'])]
    report['identical'] = (report['equal_result_fields']
                           and report['first_differing_policy_call'] is None
                           and len(first['calls']) == len(other['calls']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'fork', 'load-run', 'paired-episodes', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ppo-weights', type=Path)
    parser.add_argument('--stage', choices=('base', 'sft', 'ppo'), default='sft')
    parser.add_argument('--task', choices=POSE_TASKS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--render-samples', type=int, choices=(0, 2, 4, 8))
    args = parser.parse_args()
    if (args.repeats < 2 or not 1 <= args.max_steps <= 600
            or (args.stage == 'ppo') != bool(args.ppo_weights)):
        parser.error('Use at least two repeats, 1–600 controls, and PPO weights for the PPO stage')
    expected = paired_episodes(args.paired_episodes)[args.seed]
    if expected['task'] != args.task or expected['prompt_mode'] != 'official':
        parser.error('The pinned episode must match the task and official prompt')
    if args.render_samples is not None and expected.get('render_samples') != args.render_samples:
        parser.error('The pinned episode must use the requested render sample count')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = load_aligned_policy(args.load_run, args.base, args.fork, args.stage, args.ppo_weights)
    adapter = load_upstream_adapter(args.fork)
    records = []
    for repeat in range(args.repeats):
        started = time.monotonic()
        policy.reset_rng(args.seed)
        client = InputActionTrace(policy)
        trace_path = args.output / f'repeat-{repeat + 1}-controls.jsonl'
        for attempt in range(1, 5):
            try:
                env = open_episode(args.seed, args.task, render_samples=args.render_samples)
            except PhysicsError:
                continue
            with env, trace_path.open('w', encoding='utf-8') as control_trace:
                set_fruit_prompt(env, 'official')
                row, rollout = build_episode(env, adapter(client, replan_steps=EXECUTE), args.seed,
                                             args.max_steps, False, task=args.task, trace=control_trace,
                                             record_physics_failure=True)
                keys = ('initial_state_sha256', 'instruction', 'prompt_mode', initial_rgb_key(row, expected))
                if any(row[key] != expected[key] for key in keys):
                    rollout.close()
                    continue
                row['integration_signature'] = integration_signature(env)
                for _ in rollout:
                    pass
                control_trace.flush()
                row.update(repeat=repeat + 1, reset_attempts=attempt, stage=args.stage,
                           calls=client.calls, wall_seconds=time.monotonic() - started,
                           control_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                           **policy.provenance)
                np.savez_compressed(args.output / f'repeat-{repeat + 1}-predictions.npz',
                                    states=np.array(client.states), actions=np.array(client.actions))
                records.append(row)
                with (args.output / 'repeats.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps({key: value for key, value in row.items() if key != 'calls'}), flush=True)
                break
        else:
            raise RuntimeError(f'Could not reproduce the pinned reset for repeat {repeat + 1}')
    comparisons = [dict(repeat=record['repeat'], **compare(records[0], record)) for record in records[1:]]
    summary = dict(task=args.task, seed=args.seed, stage=args.stage, repeats=args.repeats,
                   comparisons=comparisons, all_identical=all(item['identical'] for item in comparisons))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
