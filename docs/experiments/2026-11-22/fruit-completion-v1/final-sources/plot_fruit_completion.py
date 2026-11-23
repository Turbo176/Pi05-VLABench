"""Render the verified fruit pilot rates with Wilson confidence intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    groups = [('development', 'official', 'Development\n4 starts'),
              ('ordinary', 'official', 'Held-out starts\n10 starts'),
              ('fruit_language', 'official', 'Language scenes\nOfficial prompt'),
              ('fruit_language', 'paraphrase', 'Same scenes\nParaphrased prompt'),
              ('fruit_language', 'alternate_target', 'Same scenes\nDifferent target')]
    figure, axis = plt.subplots(figsize=(11.5, 5.5), layout='constrained')
    centers = np.arange(len(groups))
    width = .24
    for index, (stage, label, color) in enumerate([
            ('base', 'Published VLABench reference', '#677384'),
            ('sft', 'Shared multitask SFT90', '#2878b5'),
            ('ppo', 'Fruit PPO1 after SFT90', '#e29b36')]):
        rows = [data['suites'][suite][mode]['stages'][stage] for suite, mode, _ in groups]
        rates = np.array([r['success_rate'] for r in rows]) * 100
        intervals = np.array([r['wilson_95'] for r in rows]) * 100
        x = centers + (index - 1) * width
        bars = axis.bar(x, rates, width, label=label, color=color, zorder=3)
        # At zero successes the Wilson lower bound can round slightly above zero.
        errors = np.maximum(0, np.vstack([rates - intervals[:, 0], intervals[:, 1] - rates]))
        axis.errorbar(x, rates, yerr=errors,
                      fmt='none', color='#263242', capsize=3, elinewidth=1, zorder=4)
        for bar, row in zip(bars, rows):
            rate = row['success_rate'] * 100
            axis.text(bar.get_x() + bar.get_width() / 2, max(3, rate / 2),
                      str(row['successes']) + '/' + str(row['trials']), ha='center', va='center',
                      color='white' if rate > 8 else '#263242', fontsize=9, fontweight='bold', zorder=5)
    axis.set_xticks(centers, [label for _, _, label in groups])
    axis.set_ylabel('Official task success (%)')
    axis.set_ylim(0, 118)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(axis='y', color='#e5e9ee', zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper left', ncol=3, fontsize=9, frameon=False)
    axis.axvline(.5, color='#c4cbd4', linestyle='--', linewidth=1)
    figure.suptitle('Pi05-VLABench: fixed fruit-first pilot', fontsize=15, fontweight='bold')
    figure.supxlabel('Bars: successes/trials. Error bars: Wilson 95% intervals. Language tests use three paired scenes.\n'
                     'Reference was already fine-tuned upstream. One PPO iteration does not establish convergence or reliable improvement.',
                     fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = []
    for suffix in ('png', 'svg'):
        path = args.output / ('success-rates.' + suffix)
        figure.savefig(path, dpi=180, facecolor='white')
        figures.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(figure)
    report = dict(comparison_sha256=hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), figures=figures)
    (args.output / 'plot-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
