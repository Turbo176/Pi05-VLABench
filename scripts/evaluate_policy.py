"""Exercise the VLABench policy evaluation loop with local smoke fixtures."""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from select_fruit import (ROBOT, camera_names, digest_arrays, open_episode,
                          physics_state, render_views, robot_state, set_fruit_prompt, success)
from vlabench_policy_io import (ACTION_HORIZON, IMAGE_SIZE, SCENE_CAMERA, WRIST_CAMERA,
                                actuator_targets, simulator_inputs)

ROOT = Path(__file__).resolve().parents[1]


def run_episode(env, infer, instruction, *, max_steps, execute_steps, initial_rgb):
    """Execute unnormalized absolute targets from infer(observation) until success or the limit."""
    if not 1 <= execute_steps <= ACTION_HORIZON or max_steps < 1:
        raise ValueError('Invalid execution length or rollout budget')
    if success(env):
        raise RuntimeError('Episode starts with the official success condition already satisfied')
    cameras = camera_names(env)
    camera_ids = {name: cameras.index(name) for name in (SCENE_CAMERA, WRIST_CAMERA)}
    outcome = 'step_budget'
    reward_sum = 0.0
    calls = steps = 0
    block_lengths = []
    while steps < max_steps:
        if steps == 0:
            rgb = {name: initial_rgb[index] for name, index in camera_ids.items()}
        else:
            rgb = {name: env.render(camera_id=index, height=IMAGE_SIZE, width=IMAGE_SIZE)
                   for name, index in camera_ids.items()}
        observation = simulator_inputs(rgb, robot_state(env), instruction)
        chunk = actuator_targets(infer(observation))
        calls += 1
        if chunk.ndim != 2 or not 1 <= len(chunk) <= ACTION_HORIZON:
            raise ValueError(f'Expected 1–{ACTION_HORIZON} actions per policy response')
        executed = 0
        for action in chunk[:min(execute_steps, max_steps - steps)]:
            timestep = env.step(action)
            steps += 1
            executed += 1
            reward_sum += float(timestep.reward)
            if success(env):
                outcome = 'success'
                break
            if timestep.last():
                outcome = 'environment_termination'
                break
        block_lengths.append(executed)
        if outcome != 'step_budget':
            break
    return {'success': outcome == 'success', 'stop_reason': outcome, 'steps': steps,
            'policy_calls': calls, 'executed_chunk_lengths': block_lengths,
            'reward_sum': reward_sum, 'environment_last': bool(timestep.last())}


def load_replay(path, execute_steps):
    with h5py.File(path, 'r') as trace:
        actions = trace['action'][:]
        expected_first_state = trace['state'][0]
        expected_final_state = trace['post_state'][-1]
        if not trace['success'][-1] or not trace['terminal'][-1]:
            raise ValueError('Replay fixture must end in official success')
    position = 0

    def infer(observation):
        nonlocal position
        if not position:
            np.testing.assert_allclose(observation['state'], expected_first_state, rtol=0, atol=1e-7)
        chunk = actions[position:position + execute_steps]
        position += len(chunk)
        return chunk

    return infer, len(actions), expected_final_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--suite', choices=('ordinary', 'fruit_language'), default='ordinary')
    parser.add_argument('--prompt-mode', default='official')
    parser.add_argument('--execute-steps', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    fixture = parser.add_mutually_exclusive_group(required=True)
    fixture.add_argument('--hold-steps', type=int, help='Apply measured joint positions and stop at this budget')
    fixture.add_argument('--replay-fixture', type=Path, help='Apply a saved successful expert action trace')
    args = parser.parse_args()

    protocol = json.loads((ROOT / 'configs/evaluation-v1.json').read_text(encoding='utf-8'))
    suite = protocol[args.suite]
    low, high = suite['seeds_inclusive']
    if ROBOT != protocol['robot'] or args.task not in suite['tasks'] or not low <= args.seed <= high:
        parser.error('Task or seed is outside the selected evaluation suite')
    if args.prompt_mode not in suite.get('prompt_modes', ['official']):
        parser.error('Prompt variant is outside the selected evaluation suite')
    if not 1 <= args.execute_steps <= ACTION_HORIZON:
        parser.error(f'--execute-steps must be between 1 and {ACTION_HORIZON}')
    max_steps = protocol['max_control_steps'] if args.hold_steps is None else args.hold_steps
    if not 1 <= max_steps <= protocol['max_control_steps']:
        parser.error('The hold budget must be positive and within the evaluation step limit')
    if args.output.exists():
        parser.error(f'Result already exists: {args.output}')

    with open_episode(args.seed, args.task) as env:
        set_fruit_prompt(env, args.prompt_mode)
        if not math.isclose(env.control_timestep(), 1 / protocol['control_hz']):
            raise RuntimeError('Simulator control frequency differs from the evaluation protocol')
        instruction = env.task.get_instruction()
        initial_rgb = render_views(env)
        initial_state_hash = digest_arrays(physics_state(env), env.physics.data.xpos, env.physics.data.xquat)
        initial_rgb_hash = digest_arrays(initial_rgb)
        if args.suite == 'fruit_language':
            prompt_config = json.loads((ROOT / 'configs/evaluation-fruit-prompts-v1.json').read_text(encoding='utf-8'))
            case = next(case for case in prompt_config['cases'] if case['seed'] == args.seed)
            prompt = case['prompts'][args.prompt_mode]
            actual = (instruction, env.task.target_entity, initial_state_hash, initial_rgb_hash)
            expected = (prompt['instruction'], prompt['target'], case['initial_state_sha256'], case['initial_rgb_sha256'])
            if actual != expected:
                raise RuntimeError('Initial scene, images, prompt, or goal differs from the frozen fruit case')
        if args.replay_fixture:
            infer, replay_steps, expected_final = load_replay(args.replay_fixture, args.execute_steps)
            source = 'saved official expert actions'
        else:
            def infer(observation):
                return np.repeat(observation['state'][None, :], args.execute_steps, axis=0)

            source = 'measured joint position hold'
        result = run_episode(env, infer, instruction, max_steps=max_steps,
                             execute_steps=args.execute_steps, initial_rgb=initial_rgb)
        if args.replay_fixture:
            if not result['success'] or result['steps'] != replay_steps or not result['environment_last']:
                raise RuntimeError('Replay did not reproduce its recorded successful termination')
            error = float(np.max(np.abs(physics_state(env) - expected_final)))
            if error > 1e-8:
                raise RuntimeError(f'Replay final physics state drifted by {error}')
            result['max_abs_final_physics_state_error'] = error
        elif result['stop_reason'] != 'step_budget' or result['steps'] != max_steps:
            raise RuntimeError('Hold smoke did not reach its configured step budget')

    report = {'scope': 'local evaluation smoke', 'action_source': source, 'protocol_version': protocol['version'],
              'task': args.task, 'seed': args.seed, 'suite': args.suite, 'prompt_mode': args.prompt_mode,
              'instruction': instruction, 'control_hz': protocol['control_hz'], 'max_control_steps': max_steps,
              'execute_steps': args.execute_steps, 'initial_state_sha256': initial_state_hash,
              'initial_rgb_sha256': initial_rgb_hash, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(f'{args.task} seed {args.seed} {args.prompt_mode}: {result["stop_reason"]} at {result["steps"]} steps',
          flush=True)


if __name__ == '__main__':
    main()
