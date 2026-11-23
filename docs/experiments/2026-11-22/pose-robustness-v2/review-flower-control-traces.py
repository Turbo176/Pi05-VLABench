"""Read completed, matched rose-42 traces; do not run or update any policy."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path('runs/pose-robustness-v2')
    candidates = {
        'published_reference': 'imitation-gate-published-reference-flower42',
        'sft_lr1e6_100': 'imitation-gate-conservative100-flower42',
        'sft_full_action_100': 'imitation-gate-full-action100-flower42',
    }
    compared, matched = {}, None
    for name, directory in candidates.items():
        gate_path = root / directory / 'gate.json'
        gate = json.loads(gate_path.read_text())
        assert gate['complete'], name
        first = gate['repeats'][0]
        contract = {key: first[key] for key in (
            'initial_state_sha256', 'initial_policy_rgb_sha256', 'instruction',
            'norm_stats_sha256', 'base_sha256', 'control_hz', 'replan_steps',
            'max_control_steps', 'render_samples', 'sampler')}
        if matched is None:
            matched = contract
        assert contract == matched, name
        trace_path = root / directory / 'repeat-1' / 'insert_flower-42-controls.jsonl'
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1, first['steps'] + 1)), name
        def first_step(predicate):
            return next((row['step'] for row in rows if predicate(row)), None)
        grasp = sum(bool(row['grasped_target']) for row in rows)
        ik = sum(not row['ik_converged'] for row in rows)
        assert grasp == first['gripper_target_contact_steps'], name
        assert ik == first['ik_nonconverged_steps'], name
        compared[name] = dict(
            controls=len(rows), success=first['success'],
            official_grasp_controls=grasp, nonconverged_ik_controls=ik,
            first_official_grasp_control=first_step(lambda row: row['grasped_target']),
            first_nonconverged_ik_control=first_step(lambda row: not row['ik_converged']),
            first_closed_gripper_command=first_step(lambda row: row['closed_command']),
            closed_gripper_command_count=sum(bool(row['closed_command']) for row in rows),
            minimum_recorded_gripper_target_distance_m=min(
                row['post_step_gripper_distance_m'] for row in rows),
            stage=('completed_goal' if first['success'] else
                   'no_official_grasp_recorded' if not grasp else 'grasp_recorded_goal_unsatisfied'),
            gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
    output = root / 'imitation-flower-followup-control-review.json'
    report = dict(kind='completed_trace_comparison_not_training_cause_identification',
                  matched_contract=matched, outcomes=compared,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policy_updates=0, additional_simulator_controls=0,
                  limitation='These traces locate observed rollout failure stages. They cannot '
                             'identify the unique data or optimization cause of SFT regression.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(compared, indent=2))


if __name__ == '__main__':
    main()
