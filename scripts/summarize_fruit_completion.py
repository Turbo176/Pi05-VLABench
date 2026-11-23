"""Validate and summarize the fixed fruit-first SFT/PPO pilot from raw episodes."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    indexed = {(r['seed'], r['prompt_mode']): r for r in rows if r['task'] == 'select_fruit'}
    if len(indexed) != sum(r['task'] == 'select_fruit' for r in rows):
        raise ValueError('Duplicated fruit episode: ' + str(path))
    return indexed


def wilson(successes, trials):
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def score(rows):
    successes = sum(r['success'] for r in rows)
    return dict(successes=successes, trials=len(rows), success_rate=successes / len(rows),
                wilson_95=wilson(successes, len(rows)),
                failure_types=dict(Counter(r['failure_type'] for r in rows if not r['success'])),
                total_controls=sum(r['steps'] for r in rows),
                ik_nonconverged_controls=sum(r['ik_nonconverged_steps'] for r in rows),
                summed_episode_wall_seconds=sum(r['wall_seconds'] for r in rows))


def paired_change(before, after, keys):
    changes = Counter()
    improved, regressed = [], []
    for key in keys:
        old, new = bool(before[key]['success']), bool(after[key]['success'])
        changes['both_success' if old and new else 'both_failure' if not old and not new
                else 'improved' if new else 'regressed'] += 1
        if new and not old:
            improved.append(list(key))
        if old and not new:
            regressed.append(list(key))
    return dict(counts=dict(changes), improved_cases=improved, regressed_cases=regressed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'reference-development', 'sft-development', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    root = args.root
    registration = json.loads((root / 'registration.json').read_text())
    handoff = json.loads((root / 'handoff.json').read_text())
    ppo = json.loads((root / 'ppo/iteration-1.json').read_text())
    if (registration['sft_sha256'] != protocol['sft_sha256']
            or handoff['checkpoint_sha256'] != protocol['sft_sha256']
            or ppo['sft_sha256'] != protocol['sft_sha256']
            or ppo['optimizer_iterations'] != 1 or ppo['actor_grad_l2'] <= 0
            or ppo['actor_delta_from_sft'] <= 0 or handoff['optimizer_updates'] != 0
            or handoff['matched_rollouts'] != 1 or handoff['matched_successes'] != 1):
        raise ValueError('Invalid parent, handoff or VLA PPO update evidence')
    changed = ppo['actor_parameter_changes']
    if any(changed[group]['changed_elements'] <= 0 for group in ('action_heads', 'action_expert_lora')):
        raise ValueError('Both action heads and action-expert LoRA must actually change')
    input_hashes = {}
    suites = {}
    episode_groups = {}
    common = ('task', 'seed', 'prompt_mode', 'instruction', 'initial_state_sha256',
              'initial_policy_rgb_sha256', 'render_samples', 'control_hz', 'replan_steps',
              'max_control_steps', 'norm_stats_sha256', 'base_sha256', 'sampler')
    for suite in ('development', 'ordinary', 'fruit_language'):
        if suite == 'development':
            paths = dict(base=args.reference_development, sft=args.sft_development,
                         ppo=root / 'development-ppo/episodes.jsonl')
            keys = {(seed, 'official') for seed in protocol[suite]['select_fruit']}
        else:
            paths = {stage: root / (suite + '-' + stage) / 'episodes.jsonl'
                     for stage in ('base', 'sft', 'ppo')}
            config = protocol[suite]
            keys = {(seed, mode) for seed in range(config['seeds_inclusive'][0],
                                                   config['seeds_inclusive'][1] + 1)
                    for mode in config.get('prompt_modes', ['official'])}
        stages = {stage: read_rows(path) for stage, path in paths.items()}
        for stage, rows in stages.items():
            if set(rows) != keys:
                raise ValueError('Incomplete or unexpected suite: ' + suite + '/' + stage)
            if stage != 'base' and any(r['sft_sha256'] != protocol['sft_sha256'] for r in rows.values()):
                raise ValueError('Evaluation used a different SFT checkpoint')
            input_hashes[suite + '/' + stage] = dict(path=str(paths[stage]), sha256=digest(paths[stage]))
        for key in sorted(keys):
            for stage in ('sft', 'ppo'):
                for field in common:
                    if stages[stage][key][field] != stages['base'][key][field]:
                        raise ValueError('Paired interface mismatch: ' + str((suite, key, stage, field)))
        modes = sorted({mode for _, mode in keys})
        suites[suite] = {}
        for mode in modes:
            selected = sorted(key for key in keys if key[1] == mode)
            suites[suite][mode] = dict(
                stages={stage: score([rows[key] for key in selected]) for stage, rows in stages.items()},
                sft_vs_reference=paired_change(stages['base'], stages['sft'], selected),
                ppo_vs_sft=paired_change(stages['sft'], stages['ppo'], selected))
        episode_groups[suite] = stages
    language = episode_groups['fruit_language']
    language_evidence = {}
    for stage, rows in language.items():
        records = []
        for seed in sorted({seed for seed, _ in rows}):
            original, paraphrase, alternate = [rows[seed, mode] for mode in
                                                ('official', 'paraphrase', 'alternate_target')]
            for other in (paraphrase, alternate):
                if any(original[field] != other[field] for field in
                       ('initial_state_sha256', 'initial_policy_rgb_sha256')):
                    raise ValueError('Language test changed the physical initial observation')
            records.append(dict(seed=seed, official_instruction=original['instruction'],
                                paraphrase=paraphrase['instruction'], alternate_instruction=alternate['instruction'],
                                official_success=original['success'], paraphrase_success=paraphrase['success'],
                                alternate_success=alternate['success'],
                                both_goal_instructions_succeeded=bool(original['success'] and alternate['success']),
                                alternate_first_action_changed=(original['first_action_sha256'] != alternate['first_action_sha256']),
                                paraphrase_first_action_changed=(original['first_action_sha256'] != paraphrase['first_action_sha256'])))
        language_evidence[stage] = records
    training_rollouts = [json.loads(line) for line in (root / 'ppo/rollouts.jsonl').read_text().splitlines()]
    if {r['seed'] for r in training_rollouts} != {1301, 1302} or len(training_rollouts) != 2:
        raise ValueError('Unexpected PPO training starts')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope='Fruit-only bounded pilot; flower and fridge are incomplete.',
                  reference='Published VLABench pi05 primitive model, already fine-tuned upstream.',
                  sft_sha256=protocol['sft_sha256'], ppo_sha256=digest(root / 'ppo/ppo-1.pt'),
                  protocol_sha256=digest(args.protocol), registration_sha256=digest(root / 'registration.json'),
                  handoff=handoff, ppo_update=ppo, ppo_training_rollouts=training_rollouts,
                  paired_interface_fields=list(common), inputs=input_hashes, suites=suites,
                  language_evidence=language_evidence,
                  limitations=[
                      'Development starts informed checkpoint selection; report them separately from held-out cases.',
                      'Ten ordinary starts and three paired language scenes are a small pilot.',
                      'One PPO iteration is a real policy update, not a convergence or reliable-improvement claim.',
                      'Changed actions under changed prompts alone do not establish correct language use.',
                      'Previously reported experiments with a different rendering profile remain separate.'
                  ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(sft_sha256=report['sft_sha256'], ppo_sha256=report['ppo_sha256'],
                         suites=suites, actor_parameter_changes=changed)), flush=True)


if __name__ == '__main__':
    main()
