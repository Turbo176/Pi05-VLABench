"""Verify the published fruit evidence on CPU; optionally render portfolio figures."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('docs/experiments/2026-11-22/fruit-completion-v1')
CONTRACT = Path('docs/experiments/2026-11-22/pose-robustness-v2/training-contract-audit.json')
STAGES = ('base', 'sft', 'ppo')
LABELS = ('Published reference', 'Shared SFT90', 'Fruit PPO1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def local(path):
    result = (ROOT / path).resolve()
    result.relative_to(ROOT)
    return result


def wilson(successes, trials):
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def verify():
    root = ROOT / EXPERIMENT
    registration = read(root / 'registration.json')
    normalized = 0
    for name, expected in registration['inputs'].items():
        data = local(name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            require(hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == expected,
                    'Registered source/config changed: ' + name)
            normalized += 1
    manifest = read(root / 'evidence-export.json')
    checked = set()
    for item in manifest['files']:
        kind, relative = item['bundle_path'].split('/', 1)
        require(kind in ('evidence', 'media'), 'Unexpected evidence bundle path')
        name = EXPERIMENT / relative if kind == 'evidence' else Path('media/policy') / relative
        path = local(name)
        require(name not in checked, 'Duplicated manifest entry: ' + str(name))
        checked.add(name)
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Evidence hash or size mismatch: ' + str(name))

    comparison = read(root / 'comparison.json')
    protocol = read(ROOT / 'configs/evaluation-fruit-completion-v1.json')
    handoff = read(root / 'handoff.json')
    ppo = read(root / 'ppo/iteration-1.json')
    saved = read(root / 'saved-checkpoint-update-audit.json')
    sft = read(root / 'sft-parent/summary.json')
    contract = read(ROOT / CONTRACT)
    require(len(rows(root / 'sft-parent/updates.jsonl')) == sft['updates'] == 90,
            'SFT update log is incomplete')
    require({registration['sft_sha256'], comparison['sft_sha256'], protocol['sft_sha256'],
             handoff['checkpoint_sha256'], ppo['sft_sha256'], saved['sft_sha256']} ==
            {protocol['sft_sha256']}, 'Different SFT parents in experiment records')
    require(comparison['ppo_sha256'] == manifest['ppo_sha256'] == saved['ppo_sha256'],
            'Different PPO checkpoint identities in records')
    require(handoff['optimizer_updates'] == 0 and handoff['matched_rollouts'] == 1
            and handoff['matched_successes'] == 1, 'Zero-update handoff did not pass')
    for measurement in handoff['measurements']:
        require(all(measurement[key] == 0 for key in ('normalized_prediction_max_error',
                    'absolute_prediction_max_error', 'replay_logprob_abs_error')),
                'Zero-update predictions or log probabilities differ')
    require(saved['all_saved_actor_changes_equal_training_report']
            and saved['actor_parameter_changes'] == ppo['actor_parameter_changes'],
            'Saved actor audit differs from training report')
    changed = ppo['actor_parameter_changes']
    require(all(changed[key]['changed_tensors'] > 0 for key in ('action_expert_lora', 'action_heads'))
            and ppo['actor_grad_l2'] > 0 and ppo['actor_delta_from_sft'] > 0,
            'Missing evidence of an effective VLA actor update')
    require(saved['optimizer_step_min'] == saved['optimizer_step_max'] == ppo['optimizer_steps'] == 19,
            'Optimizer step evidence differs')
    rollouts = rows(root / 'ppo/rollouts.jsonl')
    require(len(rollouts) == 2 and {row['seed'] for row in rollouts} == {1301, 1302}
            and sum(row['steps'] for row in rollouts) == ppo['total_controls'],
            'PPO rollout controls or starts differ')

    train = [tuple(pair) for pair in sft['train_task_seeds']]
    validation = [tuple(pair) for pair in sft['validation_task_seeds']]
    require(len(set(train)) == len(train) == 64 and len(set(validation)) == len(validation) == 5
            and not set(train).intersection(validation), 'Training/validation episode overlap')
    split = read(root / 'split-isolation.json')
    require(split['ppo_training_disjoint_from_demonstrations_and_tests'], 'PPO split audit failed')
    require(all(value['disjoint_from_training_and_validation']
                and value['language_initial_physics_and_images_match']
                for value in split['datasets'].values()), 'Initialization split audit failed')
    stats = contract['label_statistics']
    train_counts, validation_counts = Counter(task for task, _ in train), Counter(task for task, _ in validation)
    require(sum(item['frames'] for item in stats.values()) == sft['train_frames'] == 9904,
            'Training frame totals differ')
    for task, item in stats.items():
        require(item['episodes'] == train_counts[task] == sum(item['instructions'].values()),
                'Training instruction counts differ: ' + task)

    paired_fields = comparison['paired_interface_fields']
    case_records, suites, raw_count = {}, {}, 0
    for suite in ('ordinary', 'fruit_language'):
        settings = protocol[suite]
        modes = settings.get('prompt_modes', ['official'])
        keys = {(seed, mode) for seed in range(settings['seeds_inclusive'][0],
                                               settings['seeds_inclusive'][1] + 1) for mode in modes}
        stages = {}
        for stage in STAGES:
            name = root / (suite + '-' + stage) / 'episodes.jsonl'
            require(sha(name) == comparison['inputs'][suite + '/' + stage]['sha256'],
                    'Raw evaluation input hash differs')
            records = rows(name)
            indexed = {(row['seed'], row['prompt_mode']): row for row in records}
            require(len(indexed) == len(records) and set(indexed) == keys,
                    'Missing or duplicated evaluation episodes')
            raw_count += len(records)
            stages[stage] = indexed
        for key in sorted(keys):
            reference = stages['base'][key]
            for stage in ('sft', 'ppo'):
                row = stages[stage][key]
                require(all(row[field] == reference[field] for field in paired_fields),
                        'Paired evaluation interface differs: ' + str((suite, stage, key)))
                require(row['sft_sha256'] == protocol['sft_sha256'], 'Different evaluation SFT parent')
            if suite == 'fruit_language':
                for stage in STAGES:
                    original = stages[stage][key[0], 'official']
                    require(all(stages[stage][key][field] == original[field]
                                for field in ('initial_state_sha256', 'initial_policy_rgb_sha256')),
                            'Language test changes initial physics or images')
        suites[suite] = {}
        for mode in modes:
            scores = {}
            for stage in STAGES:
                selected = [stages[stage][key] for key in sorted(keys) if key[1] == mode]
                won = sum(row['success'] for row in selected)
                score = dict(successes=won, trials=len(selected), success_rate=won / len(selected),
                             wilson_95=wilson(won, len(selected)),
                             total_controls=sum(row['steps'] for row in selected))
                expected = comparison['suites'][suite][mode]['stages'][stage]
                require(all(score[key] == expected[key] for key in score), 'Aggregate score differs')
                scores[stage] = score
            suites[suite][mode] = scores
        case_records[suite] = stages

    ordinary = read(root / 'ordinary-comparison.json')
    for case in ordinary['cases']:
        for stage in STAGES:
            row = case_records['ordinary'][stage][case['seed'], 'official']
            require(all(case['outcomes'][stage][field] == row[field]
                        for field in ('success', 'steps', 'failure_type')), 'Paired case table differs')
    inputs = {str(path): sha(local(path)) for path in (
        EXPERIMENT / 'comparison.json', EXPERIMENT / 'ordinary-comparison.json',
        EXPERIMENT / 'registration.json', EXPERIMENT / 'evidence-export.json',
        EXPERIMENT / 'saved-checkpoint-update-audit.json', CONTRACT)}
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  verification=dict(registered_source_and_config_hashes=len(registration['inputs']),
                                    registered_text_crlf_normalized=normalized, evidence_files=len(checked),
                                    raw_evaluation_episodes=raw_count, paired_interface_fields=len(paired_fields),
                                    model_weights_loaded=False, simulation_executed=False),
                  input_hashes=inputs,
                  data=dict(train_episodes=len(train), validation_episodes=len(validation),
                            train_frames=sft['train_frames'], validation_frames=sft['validation_frames'],
                            training_sources=sft['training_source_counts'],
                            by_task={task: dict(training_episodes=train_counts[task],
                                               validation_episodes=validation_counts[task], frames=item['frames'],
                                               instructions=item['instructions']) for task, item in stats.items()}),
                  suites=suites, ordinary_cases=ordinary['cases'],
                  handoff=dict(measured_windows=len(handoff['measurements']),
                               normalized_prediction_max_error=max(row['normalized_prediction_max_error']
                                                                   for row in handoff['measurements']),
                               matched_closed_loop_rollouts=handoff['matched_rollouts']),
                  actor_update=changed,
                  training_cost=dict(device_reported=sft['device'],
                                     sft_update_loop_seconds=sft['training_seconds'],
                                     sft_peak_allocated_bytes=sft['peak_cuda_bytes'],
                                     ppo_rollout_update_seconds=ppo['wall_seconds'],
                                     ppo_peak_allocated_bytes=ppo['cuda_peak_allocated_bytes'],
                                     ppo_controls=ppo['total_controls'], ppo_chunks=ppo['action_chunks'],
                                     ppo_optimizer_steps=ppo['optimizer_steps']),
                  limitations=['Verifies exported records; does not reload weights or repeat a simulation.',
                               'Small fruit pilot; flower and fridge learned-policy workflows remain unfinished.',
                               'No aggregate PPO success gain; language regression remains in the full report.',
                               'Loop times exclude model loading, evaluation, diagnostics, and earlier experiments.'])
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12})
    figures = []

    def save(figure, name):
        for suffix in ('png', 'svg'):
            path = output / (name + '.' + suffix)
            figure.savefig(path, dpi=180, facecolor='white')
            figures.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), height_ratios=(1, 1.4), layout='constrained')
    rates = report['suites']['ordinary']['official']
    for index, (stage, color) in enumerate(zip(STAGES, ('#788496', '#257db0', '#d89532'))):
        row = rates[stage]
        rate, (low, high) = row['success_rate'] * 100, [x * 100 for x in row['wilson_95']]
        axes[0].barh(index, rate, color=color, height=.52)
        axes[0].errorbar(rate, index, xerr=[[max(0, rate - low)], [max(0, high - rate)]],
                         fmt='none', ecolor='#263242', capsize=4, linewidth=1.2)
        axes[0].text(rate / 2, index, f"{row['successes']}/{row['trials']}  ({rate:.0f}%)",
                     color='white', va='center', ha='center', weight='bold')
    axes[0].set(yticks=range(3), yticklabels=LABELS, xlim=(0, 100), xlabel='Official success (%)',
                title='Same ten held-out scenes | Wilson 95% intervals')
    axes[0].invert_yaxis()
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].grid(axis='x', color='#e5e9ef')
    axes[0].set_axisbelow(True)
    cases = report['ordinary_cases']
    values = np.asarray([[int(case['outcomes'][stage]['success']) for case in cases] for stage in STAGES])
    axes[1].imshow(values, cmap=ListedColormap(['#e8edf2', '#247e70']), vmin=0, vmax=1, aspect='auto')
    for row, stage in enumerate(STAGES):
        for column, case in enumerate(cases):
            item = case['outcomes'][stage]
            axes[1].text(column, row, ('Success' if item['success'] else 'Failure') + '\n' + str(item['steps']),
                         ha='center', va='center', color='white' if item['success'] else '#405264', fontsize=9)
    axes[1].set(yticks=range(3), yticklabels=LABELS, xticks=range(len(cases)),
                xticklabels=[str(case['seed']) + '\n' + case['instruction'].split()[2].capitalize()
                             for case in cases], title='Every paired outcome | numbers show executed controls')
    axes[1].tick_params(length=0, pad=8)
    axes[1].set_xticks(np.arange(-.5, len(cases), 1), minor=True)
    axes[1].set_yticks(np.arange(-.5, 3, 1), minor=True)
    axes[1].grid(which='minor', color='white', linewidth=4)
    axes[1].tick_params(which='minor', length=0)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    figure.suptitle('Fruit closed-loop evaluation | reference, SFT, and its PPO descendant',
                     fontsize=15, weight='bold')
    figure.supxlabel('10 Hz; at most 600 controls. Reference was already fine-tuned upstream.\n'
                     'SFT: 3 gains / 2 losses. PPO: 2 gains / 2 losses. No aggregate PPO gain in this sample.', fontsize=9)
    save(figure, 'paired-results')

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), width_ratios=(1, 1.6), layout='constrained')
    tasks = ('select_fruit', 'insert_flower')
    y = np.arange(2)
    for offset, field, label, color in ((-.16, 'training_episodes', 'Training', '#257db0'),
                                      (.16, 'validation_episodes', 'Validation', '#99bdcf')):
        counts = [report['data']['by_task'][task][field] for task in tasks]
        axes[0].barh(y + offset, counts, height=.3, color=color, label=label)
        for pos, count in zip(y + offset, counts):
            axes[0].text(count + .6, pos, str(count), va='center', fontsize=10)
    axes[0].set(yticks=y, yticklabels=['Fruit', 'Flower'], xlim=(0, 55), xlabel='Complete episodes',
                title='Episode split | 64 train / 5 validation')
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc='lower right', fontsize=9)
    names, counts, colors = [], [], []
    for task, color in zip(tasks, ('#257db0', '#ad789f')):
        for prompt, count in report['data']['by_task'][task]['instructions'].items():
            names.append(('Fruit: ' if task == 'select_fruit' else 'Flower: ') + prompt.split()[2])
            counts.append(count)
            colors.append(color)
    positions = np.arange(len(names))
    axes[1].barh(positions, counts, color=colors, height=.62)
    for pos, count in zip(positions, counts):
        axes[1].text(count + .2, pos, str(count), va='center')
    axes[1].set(yticks=positions, yticklabels=names, xlabel='Training episodes per literal instruction',
                xlim=(0, max(counts) + 2), title='Recorded instruction coverage')
    axes[1].invert_yaxis()
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', color='#e5e9ef')
        axis.set_axisbelow(True)
    figure.suptitle('Shared SFT dataset | 9,904 training frames', fontsize=15, weight='bold')
    figure.supxlabel('Fruit: 6,986 frames, including two expert correction episodes. Flower: 2,918 frames.\n'
                     'Validation: 744 frames. Original three-task expert gallery is a separate dataset.', fontsize=9)
    save(figure, 'dataset-coverage')
    return dict(matplotlib=matplotlib.__version__, numpy=np.__version__, figures=figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write report.json into a new directory')
    parser.add_argument('--figures', action='store_true', help='Also draw figures; requires NumPy and Matplotlib')
    args = parser.parse_args()
    if args.figures and args.output is None:
        parser.error('--figures requires --output')
    if args.output is not None and args.output.exists():
        parser.error('Output directory already exists; choose a fresh path')
    report = verify()
    report['source_sha256'] = sha(Path(__file__))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        if args.figures:
            report['plots'] = plot(report, args.output)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Verified {} evidence files and {} registered source/config hashes.'.format(
        report['verification']['evidence_files'], report['verification']['registered_source_and_config_hashes']))
    for stage, label in zip(STAGES, LABELS):
        row = report['suites']['ordinary']['official'][stage]
        print('{}: {}/{} successes, {:.0%}, {} controls'.format(
            label, row['successes'], row['trials'], row['success_rate'], row['total_controls']))
    print('Actor update: {} changed tensors; {} PPO optimizer steps; {} environment controls.'.format(
        sum(item['changed_tensors'] for item in report['actor_update'].values()),
        report['training_cost']['ppo_optimizer_steps'], report['training_cost']['ppo_controls']))
    print('Checks use saved evidence only; no GPU, weights, or simulation are required.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        print('Evidence verification failed: ' + str(error), file=sys.stderr)
        sys.exit(1)
