"""Success-rate heatmap, TwoRoom hard: CEM (LeWM + TRM cost) over its horizon × receding-horizon grid, w=1 and w=H,
next to the three LeWAM arms, same seeds and episodes, one colour scale. Mean of the 3 seeds.

Writes data/tworoom_hard/sr_heatmap_tworoom_hard.png. Usage: python analysis/tworoom_hard/plot_heatmap.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planner_comparison'))
from plot_style import INK, INK_2, SEQUENTIAL, SURFACE, style  # noqa: E402
from summary import ARMS, DATA, SEEDS, load_lewam, load_pg_cem  # noqa: E402

HORIZONS = [2, 5, 10]
RECEDING = [1, 2, 5, 10]
LEWAM_LABELS = {'constant': 'constant\n(5 blocks ahead)', 'native_d25': 'native\ndeadline 25',
                'native_d50': 'native\ndeadline 50'}


def annotate(ax, grid):
    """Value in each filled cell, dark or light ink by cell brightness."""
    for (i, j), value in np.ndenumerate(grid):
        if np.isnan(value):
            continue
        ink = SURFACE if value > 55 else INK
        ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=10, color=ink)


def cem_grid(cem, window):
    """Mean success per (h, rh) for one window; NaN where rh > h or a seed is missing."""
    grid = np.full((len(HORIZONS), len(RECEDING)), np.nan)
    runs = cem[cem.window == window].drop_duplicates(['h', 'rh', 'seed'])
    for (h, rh), cell in runs.groupby(['h', 'rh']):
        if len(cell) == len(SEEDS):
            grid[HORIZONS.index(h), RECEDING.index(rh)] = cell.sr.mean()
    return grid


def main():
    style()
    cem = load_pg_cem()
    episodes = load_lewam()
    lewam = np.array([[100 * episodes[episodes.arm == arm].groupby('seed').success.mean().mean()
                       for arm in LEWAM_LABELS]])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[4, 4, 3]),
                             constrained_layout=True)
    for ax, window, title in ((axes[0], 'w1', 'CEM, cost window 1'), (axes[1], 'wH', 'CEM, cost window H')):
        grid = cem_grid(cem, window)
        image = ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=100)
        annotate(ax, grid)
        ax.set_xticks(range(len(RECEDING)), [f'rh {r}' for r in RECEDING])
        ax.set_yticks(range(len(HORIZONS)), [f'h {h}' for h in HORIZONS])
        ax.set_title(f'{title}\n(LeWM, TRM cost, 300×10)', fontsize=10)
    ax = axes[2]
    ax.imshow(lewam, cmap=SEQUENTIAL, vmin=0, vmax=100)
    annotate(ax, lewam)
    ax.set_xticks(range(len(LEWAM_LABELS)), LEWAM_LABELS.values(), fontsize=8)
    ax.set_yticks([])
    ax.set_title('LeWAM gradient planner\n(latent MSE cost)', fontsize=10)
    for ax in axes:
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01)
    colorbar.set_label('success % (mean of 3 seeds)', color=INK_2)
    colorbar.outline.set_visible(False)
    fig.suptitle(f'TwoRoom hard: success rate, seeds {", ".join(map(str, SEEDS))}, 30 episodes each '
                 '(same episodes for every planner), budget 50 env steps', fontsize=11)
    fig.text(0.01, -0.03, 'Blank cells: rh > h. CEM cells are the stable-worldmodel cem_vs_icem_seeds25 runs on '
             'these seeds; they use the learned TRM reachability cost, LeWAM uses latent MSE, so the panels compare '
             'model + cost + planner stacks. LeWAM deadline = goal horizon given to the planner, in env steps.',
             fontsize=7, color=INK_2, ha='left', va='top', wrap=True)
    path = DATA / 'sr_heatmap_tworoom_hard.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print('wrote', path)


if __name__ == '__main__':
    main()
