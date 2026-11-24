"""Verify paired fruit outcomes and plot the complete recorded training logs."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

TEST_MANIFEST_SHA256 = '5156c7f5bdd32e3405f7a7747969b0f2faa813c30e3f498ea0f70ffead21f36d'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def verify_export(experiment):
    manifest = json.loads((experiment/'evidence/export-manifest.json').read_text(encoding='utf-8'))
    checked = 0
    for root, entries, allow_crlf in ((experiment, manifest['experiment_files'], False),
                                     (experiment.parents[3], manifest['repository_sources'], True),
                                     (experiment.parents[3], manifest['repository_media'], False)):
        for name, expected in entries.items():
            path = (root/name).resolve()
            path.relative_to(root.resolve())
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected['sha256']:
                if (not allow_crlf
                        or hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() != expected['sha256']):
                    raise ValueError('Export hash mismatch: ' + name)
            checked += 1
    return checked


def verify_training(experiment, selection, audit_path):
    evidence = experiment/'evidence'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    ppo = read(evidence/'ppo/iteration-1.json')
    audit = read(audit_path)
    warmup = read(evidence/'ppo/warmup.json')
    handoff = read(evidence/'zero-update-handoff.json')
    sft = read(evidence/'sft-parent/summary.json')
    log = [r for r in rows(evidence/'ppo/updates.jsonl') if r['iteration'] <= selection['selected_iteration']]
    applied = [r for r in log if r['optimizer_step_applied']]
    actor = [r for r in applied if r['phase'] == 'ppo']
    if (audit['sft_sha256'] != selection['sft_sha256']
            or audit['ppo_sha256'] != selection['ppo_sha256']
            or audit['audit_source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or ppo['sft_sha256'] != selection['sft_sha256']
            or handoff['checkpoint_sha256'] != selection['sft_sha256']
            or audit['optimizer_steps']['actor']['step_min'] != len(actor)
            or audit['optimizer_steps']['actor']['step_max'] != len(actor)
            or audit['optimizer_steps']['critic']['step_min'] != len(applied)
            or audit['optimizer_steps']['critic']['step_max'] != len(applied)
            or ppo['actor_optimizer_steps'] != len(actor)
            or not actor or max(r['actor_grad_l2'] for r in actor) <= 0):
        raise ValueError('Saved training, gradient or optimizer evidence differs')
    for role, independent in audit['actor_parameter_changes'].items():
        reported = ppo['actor_parameter_changes'][role]
        for key in ('tensors', 'changed_tensors', 'changed_elements'):
            if independent[key] != reported[key] or independent[key] <= 0:
                raise ValueError('Actor tensor evidence differs')
        for key in ('max_abs_delta', 'delta_l2'):
            if not math.isclose(independent[key], reported[key], rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError('Actor delta evidence differs')
    if (not warmup['actor_unchanged'] or warmup['actor_max_abs_delta'] != 0
            or warmup['critic_max_abs_delta'] <= 0 or handoff['optimizer_updates'] != 0
            or handoff['matched_rollouts'] != 1 or handoff['matched_successes'] != 1
            or sft['updates'] != 180 or len(rows(evidence/'sft-parent/updates.jsonl')) != 180):
        raise ValueError('Warmup, handoff or SFT history differs')
    for measurement in handoff['measurements']:
        if any(measurement[k] != 0 for k in ('normalized_prediction_max_error',
                                            'absolute_prediction_max_error', 'replay_logprob_abs_error')):
            raise ValueError('Zero-update handoff changed policy behavior')
    training_starts = rows(evidence/'training-starts.jsonl')
    test = rows(evidence/'test-cases.jsonl')
    development = rows(evidence/'development/cases.jsonl')
    if (set(r['initial_state_sha256'] for r in training_starts)
            & set(r['initial_state_sha256'] for r in test+development)):
        raise ValueError('RL training overlaps development or test initializations')
    rollouts = rows(evidence/'ppo/rollouts.jsonl')
    selected_cost = sum(r['steps'] for r in rollouts if r['iteration'] <= selection['selected_iteration'])
    if selected_cost != ppo['total_controls'] or selected_cost != audit['total_controls']:
        raise ValueError('Selected checkpoint interaction count differs')
    stop = read(evidence/'ppo/run-stop.json')
    partial = rows(evidence/'ppo/failed-rollout-controls.jsonl')
    completed = sum(r['steps'] for r in rollouts)+len(partial)
    if (completed != stop['total_successfully_completed_controls']
            or completed+1 != stop['total_attempted_controls']
            or len(partial) != stop['completed_controls_in_partial_episode']):
        raise ValueError('Interrupted rollout cost differs')
    repo = experiment.parents[3]
    for name, expected in (('train_fruit_ppo.py', ppo['training_script_sha256']),
                           ('ppo_pi05.py', ppo['shared_ppo_script_sha256']),
                           ('ppo_reference_aligned_fruit.py', ppo['legacy_entry_sha256']),
                           ('train_reference_aligned_fruit.py', sft['training_script_sha256'])):
        data = (repo/'scripts'/name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Executed training source differs: ' + name)
    return dict(actor_optimizer_steps=len(actor), critic_optimizer_steps=len(applied),
                checkpoint_training_controls=selected_cost, complete_run_controls=completed,
                actor_changed_tensors=sum(r['changed_tensors'] for r in audit['actor_parameter_changes'].values()))


def wilson(wins, count):
    z = 1.959963984540054
    rate = wins / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def score(episodes):
    wins = sum(row['success'] for row in episodes)
    return dict(successes=wins, trials=len(episodes), success_rate=wins / len(episodes),
                wilson_95=wilson(wins, len(episodes)),
                total_controls=sum(row['steps'] for row in episodes),
                ik_nonconverged_controls=sum(row['ik_nonconverged_steps'] for row in episodes),
                failure_types=dict(Counter(row['failure_type'] for row in episodes if not row['success'])))


def fruit(row):
    match = re.fullmatch(r'Put the (apple|banana|orange|pear) into the plate_seen', row['instruction'])
    if not match:
        raise ValueError('Unexpected official instruction')
    return match[1]


def paired(first, second):
    wins = sum(not a['success'] and b['success'] for a, b in zip(first, second))
    losses = sum(a['success'] and not b['success'] for a, b in zip(first, second))
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
            / 2 ** discordant) if discordant else 1.
    return dict(gained_scenes=wins, lost_scenes=losses, unchanged_scenes=len(first)-discordant,
                success_rate_difference=(wins-losses)/len(first), exact_mcnemar_two_sided_p=p)


def audit_checkpoints(sft_path, ppo_path, update_path):
    """Recalculate actor changes and optimizer steps from saved CPU tensors."""
    import torch

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sft = torch.load(sft_path, map_location='cpu', weights_only=True)
    ppo = torch.load(ppo_path, map_location='cpu', weights_only=True)
    report = ppo['ppo']
    if (report['sft_sha256'] != digest(sft_path)
            or report['norm_stats_sha256'] != sft['metadata']['norm_stats_sha256']
            or report['base_sha256'] != sft['metadata']['base_sha256']
            or sft['metadata']['vlm_lora_rank'] != 0):
        raise ValueError('Checkpoint parent, base, normalization or actor scope differs')
    extra = set(ppo['policy']) - set(sft['policy'])
    if (not set(sft['policy']) <= set(ppo['policy'])
            or any(not name.startswith('value_head.') for name in extra)):
        raise ValueError('Unexpected actor/critic checkpoint tensors')
    groups = {}
    for name, before in sft['policy'].items():
        after = ppo['policy'][name]
        if before.dtype != after.dtype or before.shape != after.shape:
            raise ValueError('Checkpoint parameter shape or dtype differs')
        delta = after.double() - before.double()
        role = 'action_expert_lora' if name.startswith('llm.') else 'action_heads'
        g = groups.setdefault(role, dict(tensors=0, changed_tensors=0, changed_elements=0,
                                        max_abs_delta=0., delta_squared=0.))
        changed = int(torch.count_nonzero(delta))
        g['tensors'] += 1
        g['changed_tensors'] += int(changed > 0)
        g['changed_elements'] += changed
        g['max_abs_delta'] = max(g['max_abs_delta'], float(delta.abs().max()))
        g['delta_squared'] += float(delta.square().sum())
    for role, g in groups.items():
        g['delta_l2'] = math.sqrt(g.pop('delta_squared'))
        original = report['actor_parameter_changes'][role]
        for key in ('tensors', 'changed_tensors', 'changed_elements'):
            if g[key] != original[key]:
                raise ValueError('Saved actor changes disagree with the training report')
        # Training reports float32 deltas; this independent audit subtracts in float64.
        if not math.isclose(g['max_abs_delta'], original['max_abs_delta'], rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError('Saved actor maximum change disagrees with the training report')
        if g['changed_tensors'] == 0 or not math.isclose(g['delta_l2'], original['delta_l2'], rel_tol=1e-6):
            raise ValueError('Missing or inconsistent actor parameter change')
    log = [r for r in rows(update_path) if r['iteration'] <= report['optimizer_iterations']]
    actor_steps = sum(r['phase'] == 'ppo' and r['optimizer_step_applied'] for r in log)
    critic_steps = sum(r['optimizer_step_applied'] for r in log)
    optimizer = ppo['optimizer']
    if len(optimizer['param_groups']) != 2:
        raise ValueError('Expected separate actor and critic optimizer groups')
    measured = {}
    for role, group, expected in zip(('actor', 'critic'), optimizer['param_groups'], (actor_steps, critic_steps)):
        steps = [int(optimizer['state'][index]['step']) for index in group['params']]
        if min(steps) != max(steps) or steps[0] != expected or expected <= 0:
            raise ValueError('Saved optimizer step counters disagree with applied updates')
        measured[role] = dict(parameters=len(steps), step_min=min(steps), step_max=max(steps))
    gradients = [r['actor_grad_l2'] for r in log if r['phase'] == 'ppo' and r['optimizer_step_applied']]
    if not gradients or max(gradients) <= 0 or optimizer['param_groups'][0]['params'] == []:
        raise ValueError('Missing actor gradient or optimizer evidence')
    return dict(sft_sha256=digest(sft_path), ppo_sha256=digest(ppo_path),
                audit_source_sha256=digest(Path(__file__)),
                actor_parameter_changes=groups, optimizer_steps=measured,
                max_recorded_actor_gradient=max(gradients),
                optimizer_iteration=report['optimizer_iterations'],
                total_controls=report['total_controls'], matches_training_report=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    for stage in ('base', 'sft', 'ppo'):
        parser.add_argument('--' + stage, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sft-updates', type=Path)
    parser.add_argument('--ppo-updates', type=Path)
    parser.add_argument('--sft-weights', type=Path)
    parser.add_argument('--ppo-weights', type=Path)
    parser.add_argument('--selection', type=Path, help='Defaults to final-selection.json beside the output folder')
    parser.add_argument('--verify-only', action='store_true',
                        help='Recompute outcomes and verify the saved comparison without plotting or loading weights')
    args = parser.parse_args()
    if (bool(args.sft_weights) != bool(args.ppo_weights)
            or args.sft_weights and not args.ppo_updates):
        parser.error('Saved-weight audit requires both checkpoints and the PPO update log')
    manifest = rows(args.cases)
    expected = {row['seed']: row for row in manifest}
    if (len(manifest) != 40 or len(expected) != 40
            or hashlib.sha256(args.cases.read_bytes()).hexdigest() != TEST_MANIFEST_SHA256
            or len({row['initial_state_sha256'] for row in expected.values()}) != 40):
        raise ValueError('Require forty distinct frozen test initializations')
    if {name: sum(fruit(row) == name for row in expected.values())
            for name in ('apple', 'banana', 'orange', 'pear')} != dict.fromkeys(('apple', 'banana', 'orange', 'pear'), 10):
        raise ValueError('Require ten naturally selected targets per fruit')
    keys = ('initial_state_sha256', 'initial_policy_rgb_sha256', 'instruction', 'render_samples')
    stages, hashes = {}, {}
    for stage in ('base', 'sft', 'ppo'):
        path = getattr(args, stage)
        records = rows(path)
        by_seed = {row['seed']: row for row in records}
        if len(records) != 40 or set(by_seed) != set(expected):
            raise ValueError(f'Incomplete or repeated {stage} evaluation')
        for seed, row in by_seed.items():
            if (any(row[key] != expected[seed][key] for key in keys)
                    or row['stage'] != stage or row['task'] != 'select_fruit'
                    or row['prompt_mode'] != 'official' or row['max_control_steps'] != 600
                    or row['control_hz'] != 10 or row['replan_steps'] != 5
                    or not 1 <= row['steps'] <= 600 or type(row['success']) is not bool):
                raise ValueError(f'Policy/control/initialization mismatch in {stage}:{seed}')
        stages[stage] = [by_seed[seed] for seed in sorted(expected)]
        hashes[stage] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ('base_sha256', 'norm_stats_sha256', 'sampler', 'camera_names',
                'model_render_options', 'backend', 'openpi_revision', 'reference_model_revision',
                'evaluation_script_sha256'):
        if len({json.dumps(row[key], sort_keys=True) for group in stages.values() for row in group}) != 1:
            raise ValueError(f'Inconsistent {key}')
    if len({row['sft_sha256'] for stage in ('sft', 'ppo') for row in stages[stage]}) != 1:
        raise ValueError('PPO must descend from the evaluated SFT parent')
    if len({row['ppo_sha256'] for row in stages['ppo']}) != 1:
        raise ValueError('PPO outcomes mix different checkpoints')
    report = dict(cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  episodes_sha256=hashes,
                  stages={stage: score(group) for stage, group in stages.items()},
                  by_target={name: {stage: score([r for r in group if fruit(r) == name])
                                    for stage, group in stages.items()}
                             for name in ('apple', 'banana', 'orange', 'pear')},
                  sft_vs_reference=paired(stages['base'], stages['sft']),
                  ppo_vs_sft=paired(stages['sft'], stages['ppo']))
    selection_path = args.selection or args.output.parent/'final-selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection['sft_sha256'] != stages['sft'][0]['sft_sha256']
            or selection['ppo_sha256'] != stages['ppo'][0]['ppo_sha256']
            or selection['norm_stats_sha256'] != stages['sft'][0]['norm_stats_sha256']
            or selection['test_manifest_sha256'] != report['cases_sha256']
            or selection['final_test_outcomes_consulted']):
        raise ValueError('Development selection does not match the frozen evaluated policies')
    candidates = sorted(selection['candidates'], key=lambda r: r['iteration'])
    if not candidates or any(c['trials'] != 16 for c in candidates):
        raise ValueError('Development selection requires complete sixteen-scene candidates')
    best = min(candidates, key=lambda r: (-r['successes'], r['controls'], r['iteration']))
    if (best['iteration'] != selection['selected_iteration']
            or best['ppo_sha256'] != selection['ppo_sha256']
            or best['successes'] != selection['development_successes']):
        raise ValueError('Selected checkpoint violates the recorded development rule')
    if args.verify_only:
        checked = verify_export(Path(__file__).resolve().parent)
        training = verify_training(Path(__file__).resolve().parent, selection,
                                   args.output/'saved-checkpoint-audit.json')
        saved = json.loads((args.output/'comparison.json').read_text(encoding='utf-8'))
        if saved != report:
            raise ValueError('Saved comparison differs from recomputed raw outcomes')
        print(json.dumps(dict(verified=True, verified_files=checked, paired_scenes=40, raw_episodes=120,
                             training=training, stages=report['stages']), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    audit = None
    if args.sft_weights:
        audit = audit_checkpoints(args.sft_weights, args.ppo_weights, args.ppo_updates)
        if (audit['sft_sha256'] != stages['sft'][0]['sft_sha256']
                or audit['ppo_sha256'] != stages['ppo'][0]['ppo_sha256']):
            raise ValueError('Audited checkpoint files differ from the scored policies')
        (args.output/'saved-checkpoint-audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    labels = ('Published reference', 'Shared SFT', 'PPO after same SFT')
    colors = ('#7e8a97', '#2878b5', '#e29b36')
    for i, (stage, label, color) in enumerate(zip(stages, labels, colors)):
        r = report['stages'][stage]
        lo, hi = r['wilson_95']
        axes[0].bar(i, r['success_rate']*100, color=color)
        axes[0].errorbar(i, r['success_rate']*100,
                        yerr=[[max(0., r['success_rate']-lo)*100], [max(0., hi-r['success_rate'])*100]],
                        color='#263242', capsize=4)
        axes[0].text(i, hi*100+4, f"{r['successes']}/40", ha='center')
    axes[0].set(xticks=range(3), xticklabels=('Published\nreference', 'Shared\nSFT180', 'PPO after\nsame SFT'),
                ylim=(0, 110), ylabel='Official success (%)',
                title='Forty fixed scenes · Wilson 95% intervals')
    names = list(report['by_target'])
    for i, (stage, label, color) in enumerate(zip(stages, labels, colors)):
        values = [report['by_target'][name][stage]['success_rate']*100 for name in names]
        positions = np.arange(4)+(i-1)*.25
        axes[1].bar(positions, values, width=.25, color=color, label=label)
        for position, name, value in zip(positions, names, values):
            axes[1].text(position, value+2, str(report['by_target'][name][stage]['successes']),
                         ha='center', fontsize=8)
    axes[1].set(xticks=range(4), xticklabels=names, ylim=(0, 110), ylabel='Official success (%)',
                title='Ten scenes per instructed fruit')
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis='y', color='#e5e9ef')
        ax.set_axisbelow(True)
    fig.suptitle('Pi05-VLABench · paired closed-loop evaluation', fontweight='bold')
    for extension in ('png', 'svg'):
        fig.savefig(args.output/f'success-rates.{extension}', dpi=180, facecolor='white')
    plt.close(fig)

    from matplotlib.colors import ListedColormap

    ordered = sorted(expected, key=lambda seed: (names.index(fruit(expected[seed])), seed))
    indexed = {stage: {r['seed']: r for r in group} for stage, group in stages.items()}
    outcomes = [[indexed[stage][seed]['success'] for seed in ordered] for stage in stages]
    fig, ax = plt.subplots(figsize=(12, 3.3), layout='constrained')
    ax.imshow(outcomes, cmap=ListedColormap(('#e4e9ef', '#39876c')), vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(40), xticklabels=[str(seed) for seed in ordered],
           yticks=range(3), yticklabels=('Published reference', 'Shared SFT180', 'PPO after same SFT'),
           title='Every held-out scene · green: official success · gray: failure')
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    for boundary in (9.5, 19.5, 29.5):
        ax.axvline(boundary, color='white', linewidth=3)
    for i, name in enumerate(names):
        ax.text(i*10+4.5, -.8, name.capitalize(), ha='center', fontsize=9)
    ax.set_ylim(2.5, -1.2)
    fig.savefig(args.output/'paired-scenes.png', dpi=180, facecolor='white')
    plt.close(fig)

    if selection:
        x = [0]+[c['training_controls'] for c in candidates]
        count = [selection['parent_development_successes']]+[c['successes'] for c in candidates]
        rates = np.asarray(count)/16*100
        intervals = np.asarray([wilson(k, 16) for k in count])*100
        fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
        ax.plot(x, rates, marker='o', color=colors[2], label='Same 16 development scenes')
        ax.fill_between(x, intervals[:, 0], intervals[:, 1], color=colors[2], alpha=.15,
                        label='Descriptive Wilson 95% intervals')
        for at, value, wins in zip(x, rates, count):
            ax.annotate(f'{wins}/16', (at, value), xytext=(0, 8), textcoords='offset points', ha='center')
        selected = next(i+1 for i, c in enumerate(candidates)
                        if c['iteration'] == selection['selected_iteration'])
        ax.scatter([x[selected]], [rates[selected]], marker='*', s=160, color=colors[1],
                   zorder=5, label='Selected checkpoint')
        ax.set(xlabel='PPO training controls, including critic warmup', ylabel='Official success (%)',
               ylim=(0, 110), title='Development learning curve · used for selection, not final testing')
        ax.grid(color='#e5e9ef')
        ax.legend(frameon=False, fontsize=9, loc='lower left')
        fig.savefig(args.output/'development-curve.png', dpi=180, facecolor='white')
        plt.close(fig)

    if args.sft_updates and args.ppo_updates:
        sft, ppo = rows(args.sft_updates), rows(args.ppo_updates)
        actor = [r for r in ppo if r['phase'] == 'ppo']
        warmup = [r for r in ppo if r['phase'] == 'critic_warmup']
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
        axes[0, 0].plot([r['update'] for r in sft], [r['loss'] for r in sft], color=colors[1], alpha=.65)
        axes[0, 0].set(title='Recorded SFT flow loss', xlabel='Optimizer update', ylabel='Loss')
        for phase, data, color in [('Critic warmup', warmup, '#7e8a97'), ('PPO', actor, colors[2])]:
            axes[0, 1].plot([r['total_optimizer_steps'] for r in data],
                            [r['value_loss'] for r in data], label=phase, color=color)
        axes[0, 1].set(title='Value loss on sampled training minibatches', xlabel='Optimizer step', ylabel='MSE')
        axes[0, 1].legend(frameon=False)
        axes[1, 0].plot([r['total_optimizer_steps'] for r in actor], [r['actor_loss'] for r in actor], color=colors[2])
        axes[1, 0].set(title='PPO actor surrogate loss', xlabel='Optimizer step', ylabel='Loss')
        axes[1, 1].plot([r['total_optimizer_steps'] for r in actor], [r['approx_kl'] for r in actor], color=colors[2])
        rejected = [r for r in actor if r['poststep_reverted_for_kl'] and r['postupdate_approx_kl'] is not None]
        if rejected:
            axes[1, 1].scatter([r['total_optimizer_steps'] for r in rejected],
                               [r['postupdate_approx_kl'] for r in rejected], marker='x', color='#b24b45',
                               label='Rejected post-step value', zorder=4)
        axes[1, 1].axhline(.02, linestyle='--', color='#b24b45', label='Update guard')
        axes[1, 1].set(title='Sampled executed-prefix score diagnostic', xlabel='Optimizer step', ylabel='Approximate KL statistic')
        axes[1, 1].legend(frameon=False)
        if audit:
            selected_step = max(r['total_optimizer_steps'] for r in ppo
                                if r['iteration'] <= audit['optimizer_iteration'])
            for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
                ax.axvline(selected_step, linestyle=':', color=colors[1], linewidth=1.5)
        for ax in axes.flat:
            ax.grid(color='#e5e9ef')
        fig.suptitle('Complete saved training logs · losses are not success rates', fontweight='bold')
        if audit:
            fig.supxlabel(f"Blue dotted line: selected PPO iteration {audit['optimizer_iteration']}; all saved updates are retained.")
        fig.savefig(args.output/'training.png', dpi=180, facecolor='white')
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
