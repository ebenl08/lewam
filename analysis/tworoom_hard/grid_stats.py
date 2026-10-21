"""LeWAM native (deadline 50) vs CEM on TwoRoom hard over the matched h × rh × window grid, 10 seeds × 30 episodes.

Mirrors the stable-worldmodel analyses (ported, names and maths unchanged): bootstrap mean success with 95% percentile
CI over seeds and the paired bootstrap of LeWAM − CEM over shared seeds (src/leflow/analysis/cem_vs_icem_seeds25.py),
and Welch + TOST (±5, ±10 points) on per-episode 0/1 success with Bonferroni over the 9 cells
(src/leflow/analysis/welch_tost.py). Writes to data/tworoom_hard/grid/analysis/.
Usage: python analysis/tworoom_hard/grid_stats.py
"""

import csv
import glob
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'planner_comparison'))
from plot_style import DIVERGING, INK, INK_2, SEQUENTIAL, SURFACE, style  # noqa: E402

LW = Path(__file__).resolve().parents[2]
GRID = LW / 'data' / 'tworoom_hard' / 'grid'
OUT = GRID / 'analysis'
PG_OUTPUTS = LW.parent / 'stable-worldmodel' / 'outputs' / 'cem_vs_icem_seeds25' / 'tworoom_hard'
SEEDS = [8902, 10748, 17401, 18714, 28967, 30531, 36136, 37182, 47259, 60813]
HORIZONS, RECEDING = [2, 5, 10], [1, 2, 5, 10]
CELLS = [(h, rh) for h in HORIZONS for rh in RECEDING if rh <= h]
WINDOWS = {'w1': 'cost window 1', 'wH': 'cost window H'}
ARMS = {'cem': 'CEM 300×10 (LeWM, TRM cost)', 'lewam': 'LeWAM native, deadline 50'}
BOOTSTRAPS = 10000
ALPHA = 0.05
MARGINS = (5, 10)
M_TESTS = len(CELLS)
EPISODES = re.compile(r"'episode_successes': array\(\[(.*?)\]", re.S)


def lewam_vectors(window, h, rh):
    """{seed: 0/1 episode successes} of one LeWAM cell."""
    vectors = {}
    for seed in SEEDS:
        path = GRID / f'native_d50_{window}' / f'h{h}_rh{rh}' / str(seed) / 'episodes.csv'
        if path.exists():
            vectors[seed] = pd.read_csv(path).success.to_numpy(dtype=float)
    return vectors


def cem_vectors(window, h, rh):
    """{seed: 0/1 episode successes} of one CEM cell; first folder wins when a seed repeats, as in welch_tost.py."""
    vectors = {}
    for path in sorted(glob.glob(f'{PG_OUTPUTS}/cem_{window}/h{h}_rh{rh}/*/results.txt')):
        text = Path(path).read_text()
        seed = int(re.search(r'^seed: (\d+)', text, re.M).group(1))
        if seed in SEEDS and seed not in vectors:
            values = EPISODES.search(text).group(1).replace('\n', ' ').split(',')
            vectors[seed] = np.array([float(v) for v in values if v.strip()])
    return vectors


# ---- bootstrap over seeds (cem_vs_icem_seeds25.py, cem_window_pusht_bootstrap.py)

def bootstrap_mean(rates, indices):
    return rates[indices].mean(axis=1)


def percentile_ci(samples):
    return np.percentile(samples, [2.5, 97.5])


def half_width(ci):
    """± form of a percentile CI: half its width (exact only when the CI is symmetric)."""
    return (ci[1] - ci[0]) / 2


def seed_rates(per_arm):
    """Shared seeds and {arm: SR % per shared seed}."""
    seeds = sorted(set.intersection(*[set(v) for v in per_arm.values()]))
    return seeds, {arm: np.array([100 * v[s].mean() for s in seeds]) for arm, v in per_arm.items()}


def bootstrap_cell(rng, seeds, rates):
    """Bootstrap of each arm's mean SR and of the paired difference LeWAM − CEM, one index set for all."""
    n = len(seeds)
    indices = rng.integers(0, n, (BOOTSTRAPS, n))
    boots = {arm: bootstrap_mean(rates[arm], indices) for arm in ARMS}
    boots['diff'] = bootstrap_mean(rates['lewam'] - rates['cem'], indices)
    return boots


# ---- Welch / TOST on episodes (welch_tost.py)

def pooled_scores(per_cell):
    """One score per episode: its success averaged over the cells (the episodes repeat across cells of a seed)."""
    seeds = set.intersection(*[set(v) for v in per_cell.values()])
    seeds = [s for s in sorted(seeds) if len({len(v[s]) for v in per_cell.values()}) == 1]
    return np.concatenate([np.mean([v[s] for v in per_cell.values()], axis=0) for s in seeds])


def welch(a, b):
    """Welch statistics for mean(a) - mean(b), in percentage points."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    diff = a.mean() - b.mean()
    return dict(n_a=na, n_b=nb, mean_a=100 * a.mean(), mean_b=100 * b.mean(), diff=100 * diff, se=100 * se, df=df,
                t=diff / se, p=2 * stats.t.sf(abs(diff / se), df))


def tost(res, margin):
    """Two one-sided tests of |diff| < margin (points) with the Welch standard error and df."""
    p_low = stats.t.sf((res['diff'] + margin) / res['se'], res['df'])  # H0: diff <= -margin
    p_up = stats.t.cdf((res['diff'] - margin) / res['se'], res['df'])  # H0: diff >= +margin
    return max(p_low, p_up)


def row(window, h, rh, a, b, adjust):
    res = welch(a, b)
    crit = stats.t.ppf(1 - ALPHA / 2, res['df'])
    crit_bonf = stats.t.ppf(1 - ALPHA / (2 * M_TESTS), res['df']) if adjust else float('nan')
    p_bonf = min(1.0, M_TESTS * res['p']) if adjust else float('nan')
    out = dict(window=window, h=h, rh=rh, n_lewam=res['n_a'], n_cem=res['n_b'], sr_lewam=res['mean_a'],
               sr_cem=res['mean_b'], diff=res['diff'], se=res['se'], df=res['df'], p_welch=res['p'],
               ci_low=res['diff'] - crit * res['se'], ci_high=res['diff'] + crit * res['se'], p_bonf=p_bonf,
               ci_bonf_low=res['diff'] - crit_bonf * res['se'], ci_bonf_high=res['diff'] + crit_bonf * res['se'],
               sig=res['p'] < ALPHA, sig_bonf=bool(adjust and p_bonf < ALPHA))
    for margin in MARGINS:
        p_tost = tost(res, margin)
        p_tost_bonf = min(1.0, M_TESTS * p_tost) if adjust else float('nan')
        out.update({f'tost_p_pm{margin}': p_tost, f'tost_p_bonf_pm{margin}': p_tost_bonf,
                    f'equiv_pm{margin}': p_tost < ALPHA, f'equiv_bonf_pm{margin}': bool(adjust and p_tost_bonf < ALPHA)})
    return out


def best_cell(per_cell):
    """Cell with the highest mean SR (mean over seeds of the per-seed success rate)."""
    return max(per_cell, key=lambda cell: np.mean([v.mean() for v in per_cell[cell].values()]))


def verdict(r, adjusted):
    """Welch / TOST reading of one row at ±5 points."""
    sig = r['sig_bonf'] if adjusted else r['sig']
    equiv = r['equiv_bonf_pm5'] if adjusted else r['equiv_pm5']
    if equiv:
        return 'equivalent (±5)'
    if sig:
        return 'LeWAM better' if r['diff'] > 0 else 'CEM better'
    return 'inconclusive'


# ---- figure (layout of the PG sr_bootstrap figure: rows windows, columns CEM, LeWAM, difference)

def draw_grid(ax, values, cmap, vmin, vmax, light_above):
    grid = np.full((len(HORIZONS), len(RECEDING)), np.nan)
    for (h, rh), (mean, ci) in values.items():
        i, j = HORIZONS.index(h), RECEDING.index(rh)
        grid[i, j] = mean
        ink = SURFACE if light_above(mean) else INK
        ax.text(j, i - 0.12, f'{mean:.1f}', ha='center', va='center', color=ink, fontsize=11)
        ax.text(j, i + 0.22, f'± {half_width(ci):.1f}', ha='center', va='center', color=ink, fontsize=8)
    image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(RECEDING)), [f'rh {r}' for r in RECEDING])
    ax.set_yticks(range(len(HORIZONS)), [f'h {h}' for h in HORIZONS])
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot(cells_boot, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    diff_max = max(abs(b['diff'].mean()) for b in cells_boot.values())
    diff_max = 10 * np.ceil(diff_max / 10)
    for r, window in enumerate(WINDOWS):
        for c, key in enumerate(['cem', 'lewam', 'diff']):
            values = {(h, rh): (b[key].mean(), percentile_ci(b[key])) for (w, h, rh), b in cells_boot.items()
                      if w == window}
            if key == 'diff':
                image = draw_grid(axes[r, c], values, DIVERGING, -diff_max, diff_max, lambda m: abs(m) > 0.6 * diff_max)
                title = 'LeWAM − CEM (paired over seeds)'
            else:
                image = draw_grid(axes[r, c], values, SEQUENTIAL, 0, 100, lambda m: m > 60)
                title = ARMS[key]
            axes[r, c].set_title(f'{title}\n{WINDOWS[window]}', fontsize=10, color=INK)
            if r == 1:
                colorbar = fig.colorbar(image, ax=axes[:, c], shrink=0.6, pad=0.02)
                colorbar.set_label('SR difference (points)' if key == 'diff' else 'bootstrap mean SR (%)',
                                   color=INK_2)
                colorbar.outline.set_visible(False)
    fig.suptitle('TwoRoom hard: LeWAM native (deadline 50) vs CEM, same episodes, 10 seeds × 30 episodes, '
                 'bootstrap over seeds', fontsize=12)
    fig.text(0.01, -0.02, f'Big: bootstrap mean ({BOOTSTRAPS} resamples of the 10 seeds). Small: ± half-width of the '
             '95% percentile CI. CEM = stable-worldmodel cem_vs_icem_seeds25 runs (LeWM, TRM cost); LeWAM = its own '
             'model, latent MSE cost, goal horizon counted down from 10 blocks. Blank: rh > h.', fontsize=8,
             color=INK_2, ha='left', va='top')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(r, key, spec='.1f'):
    return format(r[key], spec)


def summary_md(test_rows, boot_rows):
    lines = ['# LeWAM native (deadline 50) vs CEM on TwoRoom hard: Welch + TOST and bootstrap', '',
             f'Seeds {SEEDS}, 30 episodes each, same episodes for both arms, budget 50. Difference = LeWAM − CEM in '
             'points. Welch / TOST on per-episode 0/1 success (Bonferroni m = 9 within a window for the cell rows; '
             'pooled = each episode averaged over the 9 cells, one test; best = each arm\'s best cell, one test). '
             'Bootstrap = 10,000 resamples of seeds, paired difference over shared seeds.', '',
             '| window | cell | SR LeWAM | SR CEM | diff | Welch 95% CI | p (Bonf. for cells) | TOST ±5 p | '
             'verdict | bootstrap diff [95% CI] |', '|' + '---|' * 10]
    boot = {(b['window'], b['h'], b['rh']): b for b in boot_rows}
    for r in test_rows:
        adjusted = r['h'] not in ('all', 'best')
        cell = 'pooled (9 cells)' if r['h'] == 'all' else ('best: ' + r['rh'] if r['h'] == 'best'
                                                          else f"h{r['h']} rh{r['rh']}")
        p = r['p_bonf'] if adjusted else r['p_welch']
        p_tost = r['tost_p_bonf_pm5'] if adjusted else r['tost_p_pm5']
        b = boot.get((r['window'], r['h'], r['rh']))
        boot_text = f"{b['diff_mean']:+.1f} [{b['diff_low']:+.1f}, {b['diff_high']:+.1f}]" if b else '—'
        lines.append(f"| {r['window']} | {cell} | {fmt(r, 'sr_lewam')} | {fmt(r, 'sr_cem')} | {r['diff']:+.1f} | "
                     f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | {p:.3g} | {p_tost:.3g} | "
                     f"{verdict(r, adjusted)} | {boot_text} |")
    return '\n'.join(lines) + '\n'


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    test_rows, boot_rows, cells_boot = [], [], {}
    for window in WINDOWS:
        per_lewam, per_cem, seed_means = {}, {}, {}
        for h, rh in CELLS:
            per_arm = {'lewam': lewam_vectors(window, h, rh), 'cem': cem_vectors(window, h, rh)}
            seeds, rates = seed_rates(per_arm)
            if len(seeds) < 2:
                print(f'skip {window} h{h} rh{rh}: {len(seeds)} shared seeds')
                continue
            boots = bootstrap_cell(rng, seeds, rates)
            cells_boot[(window, h, rh)] = boots
            boot_rows.append(dict(window=window, h=h, rh=rh, n_seeds=len(seeds),
                                  **{f'{arm}_mean': boots[arm].mean() for arm in ARMS},
                                  diff_mean=boots['diff'].mean(), diff_low=percentile_ci(boots['diff'])[0],
                                  diff_high=percentile_ci(boots['diff'])[1]))
            per_lewam[(h, rh)] = {s: per_arm['lewam'][s] for s in seeds}
            per_cem[(h, rh)] = {s: per_arm['cem'][s] for s in seeds}
            seed_means[(h, rh)] = (seeds, rates)
            a, b = np.concatenate(list(per_lewam[(h, rh)].values())), np.concatenate(list(per_cem[(h, rh)].values()))
            test_rows.append(row(window, h, rh, a, b, adjust=True))
        if len(per_lewam) == len(CELLS):
            test_rows.append(row(window, 'all', 'all', pooled_scores(per_lewam), pooled_scores(per_cem),
                                 adjust=False))
            seeds = sorted(set.intersection(*[set(s) for s, _ in seed_means.values()]))
            pooled = {arm: np.mean([[r[arm][s_list.index(s)] for s in seeds] for s_list, r in seed_means.values()],
                                   axis=0) for arm in ARMS}
            boots = bootstrap_cell(rng, seeds, pooled)
            boot_rows.append(dict(window=window, h='all', rh='all', n_seeds=len(seeds),
                                  **{f'{arm}_mean': boots[arm].mean() for arm in ARMS},
                                  diff_mean=boots['diff'].mean(), diff_low=percentile_ci(boots['diff'])[0],
                                  diff_high=percentile_ci(boots['diff'])[1]))
        if per_lewam:
            cell_l, cell_c = best_cell(per_lewam), best_cell(per_cem)
            a = np.concatenate(list(per_lewam[cell_l].values()))
            b = np.concatenate(list(per_cem[cell_c].values()))
            test_rows.append(row(window, 'best', f'h{cell_l[0]}_rh{cell_l[1]} vs h{cell_c[0]}_rh{cell_c[1]}', a, b,
                                 adjust=False))
    write_csv(OUT / 'welch_tost.csv', test_rows)
    write_csv(OUT / 'sr_bootstrap_table.csv', boot_rows)
    text = summary_md(test_rows, boot_rows)
    (OUT / 'welch_tost_summary.md').write_text(text)
    if cells_boot:
        plot(cells_boot, OUT / 'sr_bootstrap_tworoom_hard_lewam_vs_cem.png')
    print(text)


if __name__ == '__main__':
    main()
