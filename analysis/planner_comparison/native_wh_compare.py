"""LeWAM native with cost window H (adaptive horizon, window clamped to the scored step) vs native w=1 and grid w=H.

Reads data/planner_comparison/p1/runs/pusht/{native_wH,native,grid_wH}/ and reuses the P1 statistics of
lewam_compare.py (Welch per cell with Bonferroni over 9, pooled, best cell, TOST ±5, seed bootstrap, McNemar).
Writes contrasts.csv, summary.md and a heatmap figure to data/planner_comparison/analysis/native_wH/.

CLI (from playground/lewam/):
    /home/eben/.conda/envs/lewam/bin/python analysis/planner_comparison/native_wh_compare.py
"""

import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lewam_compare import DATA, anomaly, cell_grid, contrast, heatmap, lewam_cell, paired_bootstrap, seed_rates, verdict
from plot_style import DIVERGING, SEQUENTIAL, style
from stats import CELLS, HEADER

OUT = DATA / 'analysis' / 'native_wH'
ARMS = ('native_wH', 'native', 'grid_wH')
TITLES = {'native_wH': 'LeWAM native w=H', 'native': 'LeWAM native w=1', 'grid_wH': 'LeWAM grid w=H'}
CONTRASTS = [('W1', 'native_wH', 'native'), ('W2', 'native_wH', 'grid_wH')]


def plot_heatmaps(arms, rng):
    """SR heatmaps of the three arms, and paired seed-bootstrap differences native_wH − each reference."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, arm in zip(axes[0], ARMS):
        rates = seed_rates(arms[arm])
        image = heatmap(ax, cell_grid({c: np.mean(list(rates[c].values())) for c in CELLS}), TITLES[arm],
                        SEQUENTIAL, 0, 100)
    fig.colorbar(image, ax=axes[0], shrink=0.8, label='SR (%)')
    axes[1, 0].axis('off')
    for ax, (_, a, b) in zip(axes[1, 1:], CONTRASTS):
        diffs = {c: paired_bootstrap({c: seed_rates(arms[a])[c]}, {c: seed_rates(arms[b])[c]}, rng)[0] for c in CELLS}
        image = heatmap(ax, cell_grid(diffs), f'{TITLES[a]} − {TITLES[b]}', DIVERGING, -20, 20)
    fig.colorbar(image, ax=axes[1], shrink=0.8, label='SR difference (pp, shared seeds)')
    fig.savefig(OUT / 'heatmaps_native_wH.png', dpi=150)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arms = {arm: {cell: lewam_cell(arm, *cell) for cell in CELLS} for arm in ARMS}

    rows, pooled = [], []
    for name, a, b in CONTRASTS:
        contrast(name, arms, a, b, rng, rows, pooled)
    with open(OUT / 'contrasts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER + ['verdict'])
        writer.writerows(rows)

    index = {name: k for k, name in enumerate(HEADER)}
    lines = ['# LeWAM native with cost window H (PushT, 25 seeds)', '', '## Seeds per cell', '',
             '| arm | seeds per cell (9 cells) |', '|---|---|']
    lines += [f'| {TITLES[arm]} | {[len(arms[arm][c]) for c in CELLS]} |' for arm in ARMS]
    lines += ['', '## Per cell (Welch, Bonferroni over 9 cells)', '',
              '| id | cell | SR a | SR b | diff | Welch 95% CI | p Bonf | TOST ±5 Bonf |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r[index['h']] in ('all', 'best'):
            continue
        lines.append(f"| {r[0]} | h{r[index['h']]}_rh{r[index['rh']]} | {r[index['mean_a']]} | {r[index['mean_b']]} "
                     f"| {r[index['diff_a_minus_b']]} | [{r[index['ci95_low']]}, {r[index['ci95_high']]}] "
                     f"| {r[index['p_bonf']]} | {r[index['equiv_bonf_pm5']]} |")
    lines += ['', '## Pooled (all 9 cells)', '',
              '| id | a − b | diff (pp) | Welch 95% CI | p | TOST ±5 p | verdict | seed bootstrap diff [95% CI] (n) '
              '| McNemar a-only / b-only, p |', '|---|---|---|---|---|---|---|---|---|']
    for item in pooled:
        r, (diff, low, high, n), (only_a, only_b, p_m) = item['pooled'], item['boot'], item['mcnemar']
        lines.append(f"| {item['contrast']} | {TITLES[item['a']]} − {TITLES[item['b']]} | {r[index['diff_a_minus_b']]} "
                     f"| [{r[index['ci95_low']]}, {r[index['ci95_high']]}] | {r[index['p_welch']]} "
                     f"| {r[index['tost_p_pm5']]} | {verdict(r)} | {diff:+.1f} [{low:+.1f}, {high:+.1f}] ({n}) "
                     f"| {only_a} / {only_b}, {p_m:.3g} |")
    lines += ['', '## Best cell of each arm', '', '| id | a − b | cells | SR a | SR b | diff | 95% CI | verdict |',
              '|---|---|---|---|---|---|---|---|']
    for item in pooled:
        r = item['best']
        lines.append(f"| {item['contrast']} | {TITLES[item['a']]} − {TITLES[item['b']]} | {r[index['rh']]} "
                     f"| {r[index['mean_a']]} | {r[index['mean_b']]} | {r[index['diff_a_minus_b']]} "
                     f"| [{r[index['ci95_low']]}, {r[index['ci95_high']]}] | {verdict(r)} |")
    lines += ['', '## h = rh anomaly (SR points, seed bootstrap 95% CI)', '', '| arm | RC(5) | RC(10) | LH |',
              '|---|---|---|---|']
    for arm in ARMS:
        values = anomaly(arms[arm], rng)
        cells = [f'{values[n][0]:+.1f} [{values[n][1]:+.1f}, {values[n][2]:+.1f}]' if n in values else '—'
                 for n in ('RC(5)', 'RC(10)', 'LH')]
        lines.append(f'| {TITLES[arm]} | ' + ' | '.join(cells) + ' |')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')

    plot_heatmaps(arms, rng)
    print('\n'.join(lines))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
