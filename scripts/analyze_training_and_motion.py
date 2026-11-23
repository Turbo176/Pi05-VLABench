"""Plot saved SFT logs and audit motion without running or changing a policy."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def trailing_mean(values, window=10):
    return np.convolve(values, np.ones(window) / window, mode='valid')


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(samples=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def phase_distribution(values, boundaries):
    return dict(all=distribution(values), at_replan=distribution(values[boundaries]),
                within_chunk=distribution(values[~boundaries]))


def quaternion_steps(quaternions):
    # Absolute dot products handle q/-q and either fixed quaternion element order.
    values = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('Invalid measured quaternion')
    values = values / norms[:, None]
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(values[1:] * values[:-1], axis=1)), 0, 1)))


def motion(rows):
    if [row['step'] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Motion trace must contain contiguous controls')
    xyz = np.asarray([r['commanded_ee_xyz'] for r in rows])
    measured = np.asarray([r['measured_ee_xyz'] for r in rows])
    rotation = quaternion_steps([r['measured_ee_quaternion'] for r in rows])
    command_delta = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
    response_delta = np.linalg.norm(np.diff(measured, axis=0), axis=1) * 1000
    command_boundary = np.asarray([r['replanned'] for r in rows[1:]], dtype=bool)
    # Measured poses are logged BEFORE each control. Row t+1 is the response to t.
    response_boundary = np.asarray([r['replanned'] for r in rows[:-1]], dtype=bool)
    raw_euler = np.linalg.norm(np.diff(np.asarray([r['commanded_euler'] for r in rows]), axis=0), axis=1)
    largest = np.argsort(command_delta)[-5:][::-1]
    metrics = dict(controls=len(rows), ik_nonconverged_controls=sum(not r['ik_converged'] for r in rows),
                   gripper_command_changes=sum(a['closed_command'] != b['closed_command']
                                               for a, b in zip(rows, rows[1:])),
                   commanded_target_step_mm=phase_distribution(command_delta, command_boundary),
                   measured_response_step_mm=phase_distribution(response_delta, response_boundary),
                   measured_response_rotation_deg=phase_distribution(rotation, response_boundary),
                   largest_raw_euler_coordinate_step_rad=float(raw_euler.max()),
                   largest_command_steps=[dict(step=rows[i + 1]['step'], at_replan=bool(command_boundary[i]),
                                               distance_mm=float(command_delta[i])) for i in largest])
    return metrics, dict(command_steps=np.arange(2, len(rows) + 1), response_controls=np.arange(1, len(rows)),
                         command_delta_mm=command_delta, response_delta_mm=response_delta,
                         response_rotation_deg=rotation)


def style(axis):
    axis.grid(axis='y', color='#e5e9ef', linewidth=.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.set_axisbelow(True)


def save(figure, output, name):
    paths = []
    for suffix in ('png', 'svg'):
        path = output / (name + '.' + suffix)
        figure.savefig(path, dpi=180, facecolor='white')
        paths.append(dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path)))
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    paths = dict(updates=root / 'sft-parent/updates.jsonl', sft_summary=root / 'sft-parent/summary.json',
                 ppo_update=root / 'ppo/iteration-1.json')
    rows = read_rows(paths['updates'])
    summary = json.loads(paths['sft_summary'].read_text())
    if len(rows) != 90 or [r['update'] for r in rows] != list(range(1, 91)) or summary['updates'] != 90:
        raise ValueError('Expected the complete registered SFT90 log')
    loss = np.asarray([r['loss'] for r in rows])
    grad = np.asarray([r['grad_l2'] for r in rows])
    if not np.isfinite(loss).all() or not np.isfinite(grad).all():
        raise ValueError('Nonfinite training signal')
    ppo = json.loads(paths['ppo_update'].read_text())
    full_rows = {}
    for stage in ('base', 'sft', 'ppo'):
        paths[stage + '_trace'] = root / ('video-' + stage) / 'select_fruit-1501-controls.jsonl'
        full_rows[stage] = read_rows(paths[stage + '_trace'])
        paths[stage + '_verification'] = root / ('video-' + stage) / 'verification.json'
        verification = json.loads(paths[stage + '_verification'].read_text())
        if len(full_rows[stage]) != verification['controls']:
            raise ValueError('Trace/video length mismatch: ' + stage)
    common = min(map(len, full_rows.values()))
    if common != 160:
        raise ValueError('Unexpected matched-scene prefix')
    measured, signals = {}, {}
    for stage, trace in full_rows.items():
        prefix, signals[stage] = motion(trace[:common])
        complete, _ = motion(trace)
        measured[stage] = dict(common_prefix=prefix, complete_rollout=complete)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'figure.titlesize': 14})
    updates = np.arange(1, len(rows) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout='constrained')
    for axis, values, title, ylabel, color in zip(
            axes, (loss, grad), ('Training objective', 'Actor gradient before clipping'),
            ('Flow-matching loss', 'Gradient L2 norm'), ('#2979b8', '#8c63b8')):
        axis.plot(updates, values, color=color, linewidth=1, alpha=.36, label='Recorded update')
        axis.scatter(updates, values, color=color, s=8, alpha=.45, zorder=3)
        axis.plot(updates[9:], trailing_mean(values), color=color, linewidth=2.3, label='10-update trailing mean')
        axis.set(title=title, xlabel='Optimizer update', ylabel=ylabel, xlim=(1, 90), ylim=(0, None))
        style(axis)
    axes[1].axhline(1, color='#bf6e3e', linestyle='--', linewidth=1, label='Clipping threshold')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle('Shared multitask SFT | 64 demonstrations, 90 updates', fontweight='bold')
    figure.supxlabel('All recorded updates are shown. Stochastic minibatch loss is not a task success rate.', fontsize=9)
    figures = save(figure, args.output, 'sft-training')
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8), sharex=True, layout='constrained')
    configs = [('command_steps', 'command_delta_mm', 'Target change (mm)', 'Commanded end-effector target'),
               ('response_controls', 'response_delta_mm', 'Movement (mm)', 'Measured translation per control'),
               ('response_controls', 'response_rotation_deg', 'Rotation (degrees)', 'Measured quaternion rotation per control')]
    for axis, (xkey, ykey, ylabel, title) in zip(axes, configs):
        for stage, label, color, alpha in [('base', 'Published reference', '#8b97a6', .55),
                                           ('sft', 'Shared SFT90', '#2979b8', .95),
                                           ('ppo', 'Fruit PPO1', '#d68a2c', .8)]:
            axis.plot(signals[stage][xkey], signals[stage][ykey], label=label, color=color,
                      linewidth=1.35, alpha=alpha)
        for start in range(6, common + 1, 5):
            axis.axvline(start, color='#c4cbd4', linewidth=.5, alpha=.4)
        axis.set(ylabel=ylabel, title=title, ylim=(0, None))
        style(axis)
    axes[0].annotate('SFT target jump: 51.3 mm\ncontrol 85, inside the chunk', xy=(85, 51.29689476108383),
                     xytext=(103, 48), fontsize=9, arrowprops=dict(arrowstyle='->', color='#334155'))
    axes[0].legend(frameon=False, loc='upper left', ncol=3, fontsize=9)
    axes[-1].set(xlabel='Control step (10 Hz)', xlim=(1, common))
    figure.suptitle('Fruit 1501 | motion audit on the common first 160 controls', fontweight='bold')
    figure.supxlabel('Vertical marks: replanning every five controls. Measured response is aligned to the preceding command.\n'
                     'One scene only; the reference continued to 600 controls. No smoothing or replay edits were applied.', fontsize=9)
    figures += save(figure, args.output, 'motion-audit')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), scope='Saved-log analysis; no policy runs or updates.',
                  source_sha256=digest(Path(__file__)),
                  inputs={name: dict(path=path.as_posix(), sha256=digest(path)) for name, path in paths.items()},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
                  sft=dict(updates=len(rows), first_loss=float(loss[0]), last_loss=float(loss[-1]),
                           first_10_mean=float(loss[:10].mean()), last_10_mean=float(loss[-10:].mean()),
                           loss=distribution(loss), gradient_before_clipping=distribution(grad),
                           smoothing='10-update trailing mean; raw updates retained',
                           validation_loss_history_available=False),
                  ppo=dict(iterations=ppo['optimizer_iterations'], optimizer_steps=ppo['optimizer_steps'],
                           controls=ppo['total_controls'], actor_critic_loss_history_available=False),
                  motion=dict(scene_seed=1501, common_prefix_controls=common, stages=measured,
                              measured_pose_timing='pre-control; row t+1 minus row t is the physical response to control t',
                              orientation_metric='2 * acos(abs(dot(normalized measured quaternions))), degrees',
                              limitations=['Single scene; descriptive measurements do not establish a unique jitter cause.',
                                           'Raw Euler coordinate jumps can be angle wrapping rather than physical rotation.',
                                           'No simulator substep, latency, or joint-velocity trace is available.']), figures=figures)
    (args.output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), sft_updates=len(rows), trace_prefix=common,
                          figures=[f['path'] for f in figures])), flush=True)


if __name__ == '__main__':
    main()
