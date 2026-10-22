"""Success-rate heatmap, Puzzle 3x3 paper reproduction: arm x seed, then our mean next to the paper's.

Writes data/paper_repro/sr_heatmap_puzzle.png. Usage: python analysis/paper_repro/plot_heatmap.py
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planner_comparison'))
from plot_style import INK, INK_2, SEQUENTIAL, SURFACE, style  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / 'data' / 'paper_repro'
SEEDS = [42, 0, 1]
ARMS = {'reactive': 'reactive policy', 'best_of_k': 'best-of-32', 'grad': 'gradient planner'}
PAPER = {'reactive': (76.7, 8.1), 'best_of_k': (76.7, 10.1), 'grad': (78.7, 9.5)}  # docs/REPRODUCE.md


def success_rate(task, arm, seed):
    text = (DATA / task / arm / str(seed) / 'results.txt').read_text()
    return float(re.search(r"'success_rate': ([0-9.]+)", text).group(1))


def annotate(ax, grid, labels):
    """Label in each cell, dark or light ink by cell brightness."""
    for (i, j), value in np.ndenumerate(grid):
        ink = SURFACE if value > 55 else INK
        ax.text(j, i, labels[i][j], ha='center', va='center', fontsize=10, color=ink)


def finish(ax):
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    style()
    per_seed = np.array([[success_rate('puzzle', arm, seed) for seed in SEEDS] for arm in ARMS])
    means = per_seed.mean(axis=1)
    stds = per_seed.std(axis=1, ddof=1)
    paper = np.array([PAPER[arm] for arm in ARMS])
    summary = np.column_stack([means, paper[:, 0]])
    summary_labels = [[f'{m:.1f} ± {s:.1f}', f'{pm:.1f} ± {ps:.1f}']
                      for m, s, (pm, ps) in zip(means, stds, paper)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=[3, 2]),
                             constrained_layout=True)
    ax = axes[0]
    image = ax.imshow(per_seed, cmap=SEQUENTIAL, vmin=0, vmax=100)
    annotate(ax, per_seed, [[f'{v:.0f}' for v in row] for row in per_seed])
    ax.set_xticks(range(len(SEEDS)), [f'seed {s}' for s in SEEDS])
    ax.set_yticks(range(len(ARMS)), ARMS.values())
    ax.set_title('ours, per seed (50 episodes each)', fontsize=10)
    ax = axes[1]
    ax.imshow(summary, cmap=SEQUENTIAL, vmin=0, vmax=100)
    annotate(ax, summary, summary_labels)
    ax.set_xticks(range(2), ['ours', 'paper'])
    ax.set_yticks([])
    ax.set_title('mean ± std over seeds', fontsize=10)
    for ax in axes:
        finish(ax)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01)
    colorbar.set_label('success %', color=INK_2)
    colorbar.outline.set_visible(False)
    fig.suptitle('OGBench Puzzle 3x3, released LeWAM checkpoint: goal-reaching success '
                 '(goal offset 25, budget 50)', fontsize=11)
    fig.text(0.01, -0.03, 'Paper = docs/REPRODUCE.md table, seeds 42, 0, 1. Std is the sample std (ddof = 1). '
             'Same episodes for every arm within a seed.', fontsize=7, color=INK_2, ha='left', va='top')
    path = DATA / 'sr_heatmap_puzzle.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print('wrote', path)


if __name__ == '__main__':
    main()
