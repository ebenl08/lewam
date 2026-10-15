"""P1 analysis: LeWAM gradient planner on PushT vs CEM / iCEM faithful / LeFlow-prior arms, and the h = rh anomaly.

Reads LeWAM runs from data/planner_comparison/p1/ and the stable-worldmodel baselines read-only from its outputs/.
Writes CSVs, a markdown summary and plots to data/planner_comparison/analysis/p1/.
- P1a reproduction table (paper cell, 3 envs, exact binomial CI).
- Contrasts B1-B5 (Welch per cell with Bonferroni over 9, pooled, best cell; TOST ±5; four-way verdict), plus a paired
  seed bootstrap and an exact McNemar test on the pooled rows.
- Anomaly indices RC(5), RC(10), LH per arm, seed bootstrap CI.
- Plots: P-E1 heatmaps + difference panels, P-N2 TOST forest, P-N3 receding-horizon profile, P-N4 anomaly bars.

CLI (from playground/lewam/):
    /home/eben/.conda/envs/lewam/bin/python analysis/planner_comparison/lewam_compare.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from stats import (ALPHA, BOOTSTRAPS, CELLS, HEADER, best_row, episode_vectors, percentile_ci, pooled_scores, row)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data' / 'planner_comparison'
OUT = DATA / 'analysis' / 'p1'
PG_OUTPUTS = REPO.parent / 'stable-worldmodel' / 'outputs'
MARGIN = 5
PAPER = {'pusht': 98.7, 'tworoom': 100.0, 'reacher': 100.0}
HORIZONS, RECEDING = (2, 5, 10), (1, 2, 5, 10)

# baseline folders, as PG src/leflow/analysis/cem_vs_icem_seeds25.py and heldout_pusht.py (first folder wins)
OLD_FOLDERS = {
    ('cem', 'w1'): ['plan_config_cem'],
    ('cem', 'wH'): ['plan_config_cem_windowed-vary_latent-drift'],
    ('icem_faithful', 'w1'): ['plan_config_icem2_faithful', 'plan_config_icem2_latent-drift_faithful'],
    ('icem_faithful', 'wH'): ['plan_config_icem2_windowed-vary_faithful'],
}
LEFLOW_DIRS = {'leflow_cem': 'leflow_warm_k16_n64_clamp_fitted_cem_s300x5',
               'leflow_icem': 'leflow_warm_k16_n64_clamp_fitted_s300x5'}
LEFLOW_OBJECTIVE = {'w1': 'rollout_goal', 'wH': 'wmse'}
TITLES = {'native': 'LeWAM native', 'grid_w1': 'LeWAM grid w=1', 'grid_wH': 'LeWAM grid w=H',
          'cem': 'CEM', 'icem_faithful': 'iCEM faithful', 'leflow_cem': 'LeFlow → CEM', 'leflow_icem': 'LeFlow → iCEM'}


def lewam_cell(arm, h, rh):
    return episode_vectors([f'{DATA}/p1/runs/pusht/{arm}/h{h}_rh{rh}/*/'])


def baseline_cell(arm, window, h, rh):
    if arm in LEFLOW_DIRS:
        return episode_vectors([f'{PG_OUTPUTS}/icem_prior/pusht/{LEFLOW_DIRS[arm]}/{LEFLOW_OBJECTIVE[window]}/h{h}_rh{rh}/*/'])
    patterns = [f'{PG_OUTPUTS}/solver_sweep/{folder}/pusht_mse/h{h}_rh{rh}/*/' for folder in OLD_FOLDERS[(arm, window)]]
    patterns.append(f'{PG_OUTPUTS}/cem_vs_icem_seeds25/pusht/{arm}_{window}/h{h}_rh{rh}/*/')
    return episode_vectors(patterns)


def load_arms():
    """{arm label: {(h, rh): {seed: 0/1 episodes}}} for the 3 LeWAM arms and the 8 baseline arms."""
    arms = {arm: {cell: lewam_cell(arm, *cell) for cell in CELLS} for arm in ('native', 'grid_w1', 'grid_wH')}
    for base in ('cem', 'icem_faithful', 'leflow_cem', 'leflow_icem'):
        for window in ('w1', 'wH'):
            arms[f'{base}_{window}'] = {cell: baseline_cell(base, window, *cell) for cell in CELLS}
    return arms


def label(arm):
    for window, text in (('_w1', ' w=1'), ('_wH', ' w=H')):
        if arm.endswith(window) and not arm.startswith('grid'):
            return TITLES[arm[:-3]] + text
    return TITLES[arm]


# ---------------------------------------------------------------- statistics

def seed_rates(per_cell):
    """{cell: {seed: SR %}}."""
    return {cell: {s: 100 * v.mean() for s, v in seeds.items()} for cell, seeds in per_cell.items()}


def paired_bootstrap(rates_a, rates_b, rng):
    """Mean over cells of SR(a) - SR(b), per shared seed, resampled over seeds: (diff, low, high, n seeds)."""
    seeds = sorted(set.intersection(*[set(r[c]) for r in (rates_a, rates_b) for c in rates_a]))
    diffs = np.array([np.mean([rates_a[c][s] - rates_b[c][s] for c in rates_a]) for s in seeds])
    draws = diffs[rng.integers(0, len(seeds), (BOOTSTRAPS, len(seeds)))].mean(axis=1)
    low, high = percentile_ci(draws)
    return diffs.mean(), low, high, len(seeds)


def mcnemar(per_a, per_b):
    """Exact McNemar on episode pairs (same seed, same episode index, all cells); seeds with unequal episode counts
    are skipped. Returns (a only, b only, p)."""
    only_a = only_b = 0
    for cell in per_a:
        for seed in set(per_a[cell]) & set(per_b[cell]):
            a, b = per_a[cell][seed], per_b[cell][seed]
            if len(a) != len(b):
                continue
            only_a += int(((a == 1) & (b == 0)).sum())
            only_b += int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = scipy_stats.binomtest(only_a, n).pvalue if n else 1.0
    return only_a, only_b, p


def verdict(out_row):
    """Four-way verdict (Lakens 2017) from the Welch 95% CI and TOST at ±MARGIN."""
    index = {name: k for k, name in enumerate(HEADER)}
    different = out_row[index['sig_uncorrected']]
    equivalent = out_row[index[f'equiv_pm{MARGIN}']]
    if different and equivalent:
        return 'equivalent, different'
    if different:
        return 'better' if float(out_row[index['diff_a_minus_b']]) > 0 else 'worse'
    return 'equivalent' if equivalent else 'inconclusive'


def contrast(name, arms, a, b, rng, rows, pooled):
    """Per-cell, pooled and best-cell rows for arm a - arm b; pooled extras go to `pooled`."""
    per_a, per_b = {}, {}
    for cell in CELLS:
        if len(arms[a][cell]) < 2 or len(arms[b][cell]) < 2:
            continue
        per_a[cell], per_b[cell] = arms[a][cell], arms[b][cell]
        x, y = np.concatenate(list(per_a[cell].values())), np.concatenate(list(per_b[cell].values()))
        rows.append(row(name, 'pusht', f'{a} - {b}', cell[0], cell[1], x, y, adjust=True) + [''])
    if len(per_a) < len(CELLS):
        print(f'skip pooled {name}: {len(per_a)} of {len(CELLS)} cells')
        return
    pooled_row = row(name, 'pusht', f'{a} - {b}', 'all', 'all', pooled_scores(per_a), pooled_scores(per_b), adjust=False)
    rows.append(pooled_row + [verdict(pooled_row)])
    best = best_row(name, 'pusht', f'{a} - {b}', per_a, per_b)
    rows.append(best + [verdict(best)])
    boot = paired_bootstrap(seed_rates(per_a), seed_rates(per_b), rng)
    only_a, only_b, p_mcnemar = mcnemar(per_a, per_b)
    pooled.append(dict(contrast=name, a=a, b=b, pooled=pooled_row, best=best, boot=boot,
                       mcnemar=(only_a, only_b, p_mcnemar)))


def anomaly(per_cell, rng):
    """RC(5), RC(10), LH in SR points with seed bootstrap 95% CI: {index: (value, low, high)}."""
    rates = seed_rates(per_cell)
    definitions = {'RC(5)': ((5, 5), (5, 1)), 'RC(10)': ((10, 10), (10, 1)), 'LH': ((10, 1), (2, 1))}
    out = {}
    for name, (plus, minus) in definitions.items():
        seeds = sorted(set(rates[plus]) & set(rates[minus]))
        if len(seeds) < 2:
            continue
        diffs = np.array([rates[plus][s] - rates[minus][s] for s in seeds])
        draws = diffs[rng.integers(0, len(seeds), (BOOTSTRAPS, len(seeds)))].mean(axis=1)
        out[name] = (diffs.mean(), *percentile_ci(draws))
    return out


# ---------------------------------------------------------------- reproduction

def reproduction_table():
    lines = ['| env | seeds | successes | SR | exact 95% CI | paper | paper inside CI |', '|---|---|---|---|---|---|---|']
    for env, paper in PAPER.items():
        vectors = episode_vectors([f'{DATA}/p1/repro/{env}/*/'])
        if not vectors:
            continue
        k = int(sum(v.sum() for v in vectors.values()))
        n = sum(len(v) for v in vectors.values())
        ci = scipy_stats.binomtest(k, n).proportion_ci(0.95, 'exact')
        per_seed = ' / '.join(f'{100 * vectors[s].mean():.0f}' for s in sorted(vectors))
        inside = 100 * ci.low <= paper <= 100 * ci.high
        lines.append(f'| {env} | {per_seed} | {k}/{n} | {100 * k / n:.1f} | [{100 * ci.low:.1f}, {100 * ci.high:.1f}] '
                     f'| {paper} | {inside} |')
    return lines


# ---------------------------------------------------------------- plots

def cell_grid(values):
    """(3, 4) array over h x rh, NaN where rh > h."""
    grid = np.full((len(HORIZONS), len(RECEDING)), np.nan)
    for (h, rh), value in values.items():
        grid[HORIZONS.index(h), RECEDING.index(rh)] = value
    return grid


def heatmap(ax, grid, title, cmap, vmin, vmax):
    image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(len(RECEDING)), [f'rh{r}' for r in RECEDING])
    ax.set_yticks(range(len(HORIZONS)), [f'h{h}' for h in HORIZONS])
    ax.set_title(title, fontsize=10)
    return image


def plot_heatmaps(arms, rng):
    """P-E1: per window, SR heatmaps (LeWAM arm + 4 baselines) and paired-bootstrap differences LeWAM - baseline."""
    panels = {'w1': ('grid_w1', ['cem_w1', 'icem_faithful_w1', 'leflow_cem_w1', 'leflow_icem_w1']),
              'wH': ('grid_wH', ['cem_wH', 'icem_faithful_wH', 'leflow_cem_wH', 'leflow_icem_wH']),
              'native': ('native', ['grid_w1', 'grid_wH', 'cem_w1', 'cem_wH'])}
    for window, (lewam, baselines) in panels.items():
        fig, axes = plt.subplots(2, 1 + len(baselines), figsize=(4 * (1 + len(baselines)), 7), constrained_layout=True)
        mean_sr = lambda arm: {c: np.mean(list(seed_rates(arms[arm])[c].values())) for c in CELLS if arms[arm][c]}
        for ax, arm in zip(axes[0], [lewam] + baselines):
            image = heatmap(ax, cell_grid(mean_sr(arm)), label(arm), 'viridis', 0, 100)
        fig.colorbar(image, ax=axes[0], shrink=0.8, label='SR (%)')
        axes[1, 0].axis('off')
        for ax, arm in zip(axes[1, 1:], baselines):
            diffs = {}
            for cell in CELLS:
                if len(arms[lewam][cell]) > 1 and len(arms[arm][cell]) > 1:
                    diffs[cell] = paired_bootstrap({cell: seed_rates(arms[lewam])[cell]},
                                                   {cell: seed_rates(arms[arm])[cell]}, rng)[0]
            image = heatmap(ax, cell_grid(diffs), f'{label(lewam)} − {label(arm)}', 'RdBu', -60, 60)
        fig.colorbar(image, ax=axes[1], shrink=0.8, label='SR difference (pp, shared seeds)')
        fig.savefig(OUT / f'pe1_heatmaps_{window}.png', dpi=150)
        plt.close(fig)


def plot_forest(pooled):
    """P-N2: pooled and best-cell differences, 90% (thick) and 95% (thin) Welch CI, ±MARGIN band."""
    index = {name: k for k, name in enumerate(HEADER)}
    entries = []
    for item in pooled:
        for kind in ('pooled', 'best'):
            r = item[kind]
            diff, se, df = float(r[index['diff_a_minus_b']]), float(r[index['se']]), float(r[index['welch_df']])
            entries.append((f"{label(item['a'])} − {label(item['b'])}, {kind}", diff, se, df, verdict(r)))
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(entries) + 1), constrained_layout=True)
    ax.axvspan(-MARGIN, MARGIN, color='0.9')
    ax.axvline(0, color='0.5', lw=0.8)
    for y, (name, diff, se, df, text) in enumerate(entries):
        for level, width in ((0.95, 1), (0.90, 3)):
            half = scipy_stats.t.ppf(1 - (1 - level) / 2, df) * se
            ax.plot([diff - half, diff + half], [y, y], color='C0', lw=width)
        ax.plot(diff, y, 'o', color='C3')
        ax.text(1.01, y, text, transform=ax.get_yaxis_transform(), va='center', fontsize=8)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('SR difference a − b (pp)')
    fig.savefig(OUT / 'pn2_forest.png', dpi=150)
    plt.close(fig)


def plot_profiles(arms):
    """P-N3: SR vs rh at h = 5 and h = 10, one line per planner, bootstrap-free mean ± 95% t CI over seeds."""
    planners = ['native', 'grid_w1', 'grid_wH', 'cem_w1', 'cem_wH', 'leflow_cem_w1', 'leflow_cem_wH']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    for ax, h in zip(axes, (5, 10)):
        receding = [rh for rh in RECEDING if rh <= h]
        for arm in planners:
            means, halves = [], []
            for rh in receding:
                rates = np.array(list(seed_rates(arms[arm])[(h, rh)].values()))
                means.append(rates.mean() if len(rates) else np.nan)
                halves.append(scipy_stats.t.ppf(0.975, len(rates) - 1) * rates.std(ddof=1) / np.sqrt(len(rates))
                              if len(rates) > 1 else np.nan)
            style = '-' if arm in ('native', 'grid_w1', 'grid_wH') else '--'
            ax.errorbar(range(len(receding)), means, yerr=halves, fmt=style + 'o', capsize=3, label=label(arm))
        ax.set_xticks(range(len(receding)), [f'rh{r}' for r in receding])
        ax.set_title(f'h = {h}')
        ax.set_ylim(0, 105)
    axes[0].set_ylabel('SR (%)')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.savefig(OUT / 'pn3_rh_profile.png', dpi=150)
    plt.close(fig)


def plot_anomaly(indices):
    """P-N4: RC(5), RC(10), LH per arm with 95% bootstrap CI."""
    names = ['RC(5)', 'RC(10)', 'LH']
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    width = 0.8 / len(indices)
    for k, (arm, values) in enumerate(indices.items()):
        xs = np.arange(len(names)) + k * width - 0.4 + width / 2
        mean = [values.get(n, (np.nan,) * 3)[0] for n in names]
        low = [values.get(n, (np.nan,) * 3)[1] for n in names]
        high = [values.get(n, (np.nan,) * 3)[2] for n in names]
        ax.bar(xs, mean, width, label=label(arm),
               yerr=[np.subtract(mean, low), np.subtract(high, mean)], capsize=2)
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel('SR points')
    ax.legend(fontsize=7, ncol=4)
    fig.savefig(OUT / 'pn4_anomaly.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

CONTRASTS = [
    ('B1', 'grid_w1', 'cem_w1'), ('B1', 'grid_w1', 'icem_faithful_w1'),
    ('B2', 'grid_wH', 'cem_wH'), ('B2', 'grid_wH', 'icem_faithful_wH'),
    ('B3', 'grid_w1', 'leflow_cem_w1'), ('B3', 'grid_w1', 'leflow_icem_w1'),
    ('B3', 'grid_wH', 'leflow_cem_wH'), ('B3', 'grid_wH', 'leflow_icem_wH'),
    ('B4', 'native', 'grid_w1'), ('B4', 'native', 'grid_wH'),
]
B5_BASELINES = ['cem_w1', 'cem_wH', 'icem_faithful_w1', 'icem_faithful_wH',
                'leflow_cem_w1', 'leflow_cem_wH', 'leflow_icem_w1', 'leflow_icem_wH']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arms = load_arms()
    counts = {arm: [len(arms[arm][c]) for c in CELLS] for arm in arms}

    rows, pooled = [], []
    for name, a, b in CONTRASTS:
        contrast(name, arms, a, b, rng, rows, pooled)
    paper_cell = np.concatenate(list(arms['native'][(5, 1)].values()))
    for base in B5_BASELINES:
        best = max(CELLS, key=lambda c: np.mean([v.mean() for v in arms[base][c].values()]))
        other = np.concatenate(list(arms[base][best].values()))
        b5 = row('B5', 'pusht', f'native h5_rh1 - {base} best', 'paper', f'h{best[0]}_rh{best[1]}', paper_cell, other,
                 adjust=False)
        rows.append(b5 + [verdict(b5)])
    with open(OUT / 'contrasts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER + ['verdict'])
        writer.writerows(rows)

    indices = {arm: anomaly(arms[arm], rng) for arm in
               ['native', 'grid_w1', 'grid_wH', 'cem_w1', 'cem_wH', 'icem_faithful_w1', 'icem_faithful_wH',
                'leflow_cem_w1', 'leflow_cem_wH', 'leflow_icem_w1', 'leflow_icem_wH']}

    index = {name: k for k, name in enumerate(HEADER)}
    lines = ['# P1 LeWAM planner comparison: summary', '', '## P1a reproduction (paper cell)', ''] + reproduction_table()
    lines += ['', '## Seeds per cell', '', '| arm | seeds per cell (9 cells) |', '|---|---|']
    lines += [f'| {label(arm)} | {counts[arm]} |' for arm in arms]
    lines += ['', '## Pooled contrasts (all 9 cells)', '',
              '| id | a − b | diff (pp) | Welch 95% CI | p | TOST ±5 p | verdict | seed bootstrap diff [95% CI] (n) '
              '| McNemar a-only / b-only, p |', '|---|---|---|---|---|---|---|---|---|']
    for item in pooled:
        r, (diff, low, high, n), (only_a, only_b, p_m) = item['pooled'], item['boot'], item['mcnemar']
        lines.append(f"| {item['contrast']} | {label(item['a'])} − {label(item['b'])} | {r[index['diff_a_minus_b']]} "
                     f"| [{r[index['ci95_low']]}, {r[index['ci95_high']]}] | {r[index['p_welch']]} "
                     f"| {r[index['tost_p_pm5']]} | {verdict(r)} | {diff:+.1f} [{low:+.1f}, {high:+.1f}] ({n}) "
                     f"| {only_a} / {only_b}, {p_m:.3g} |")
    lines += ['', '## Best cell of each arm', '', '| id | a − b | cells | SR a | SR b | diff | 95% CI | verdict |',
              '|---|---|---|---|---|---|---|---|']
    for item in pooled:
        r = item['best']
        lines.append(f"| {item['contrast']} | {label(item['a'])} − {label(item['b'])} | {r[index['rh']]} "
                     f"| {r[index['mean_a']]} | {r[index['mean_b']]} | {r[index['diff_a_minus_b']]} "
                     f"| [{r[index['ci95_low']]}, {r[index['ci95_high']]}] | {verdict(r)} |")
    lines += ['', '## B5: native paper cell h5_rh1 vs each baseline best cell', '',
              '| baseline (best cell) | SR native | SR baseline | diff | 95% CI | verdict |', '|---|---|---|---|---|---|']
    for r in rows:
        if r[0] == 'B5':
            lines.append(f"| {r[2].split(' - ')[1]} ({r[4]}) | {r[index['mean_a']]} | {r[index['mean_b']]} "
                         f"| {r[index['diff_a_minus_b']]} | [{r[index['ci95_low']]}, {r[index['ci95_high']]}] | {r[-1]} |")
    lines += ['', '## h = rh anomaly (SR points, seed bootstrap 95% CI)', '',
              'RC(h) = SR(h, rh=h) − SR(h, rh=1); LH = SR(h10, rh1) − SR(h2, rh1).', '',
              '| arm | RC(5) | RC(10) | LH |', '|---|---|---|---|']
    for arm, values in indices.items():
        cells = [f'{values[n][0]:+.1f} [{values[n][1]:+.1f}, {values[n][2]:+.1f}]' if n in values else '—'
                 for n in ('RC(5)', 'RC(10)', 'LH')]
        lines.append(f'| {label(arm)} | ' + ' | '.join(cells) + ' |')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')

    plot_heatmaps(arms, rng)
    plot_forest(pooled)
    plot_profiles(arms)
    plot_anomaly(indices)
    print('\n'.join(lines))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
