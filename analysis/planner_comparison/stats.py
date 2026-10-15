"""Statistics shared by the LeWAM planner comparison: per-episode loader, Welch, TOST, best cell.

Copies of playground/stable-worldmodel/src/leflow/analysis/welch_tost.py (episode_vectors, welch, tost, row, HEADER)
and best_cell.py (cell_matrix, best_cell_stats), and percentile_ci of cem_window_pusht_bootstrap.py. Same names and
logic; no import from that project, so its runs are only read.
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ALPHA = 0.05
MARGINS = (5, 10)  # equivalence margins in success-rate points; ±5 is the planned one
CELLS = [(h, rh) for h in (2, 5, 10) for rh in (1, 2, 5, 10) if rh <= h]
M_TESTS = len(CELLS)
BOOTSTRAPS = 10000
EPISODES = re.compile(r"'episode_successes': array\(\[(.*?)\]", re.S)


def episode_vectors(patterns):
    """{seed: 0/1 array of episode successes in env order}, first folder wins when a seed repeats."""
    vectors = {}
    for pattern in patterns:
        for folder in sorted(glob.glob(pattern)):
            config = Path(folder) / 'config.json'
            results = glob.glob(f'{folder}/*results*.txt')
            if not config.exists() or not results:
                continue
            match = EPISODES.search(open(results[0]).read())
            if match:
                tokens = [token.strip() for token in match.group(1).split(',') if token.strip()]
                flags = [token == 'True' or (token != 'False' and float(token) > 0.5) for token in tokens]  # bool or 0./1.
                vectors.setdefault(int(json.load(open(config))['seed']), np.array(flags, dtype=float))
    return vectors


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


def row(comparison, env, window, h, rh, a, b, adjust):
    res = welch(a, b)
    crit = stats.t.ppf(1 - ALPHA / 2, res['df'])
    crit_bonf = stats.t.ppf(1 - ALPHA / (2 * M_TESTS), res['df']) if adjust else float('nan')
    p_bonf = min(1.0, M_TESTS * res['p']) if adjust else float('nan')
    out = [comparison, env, window, h, rh, res['n_a'], res['n_b'], f"{res['mean_a']:.2f}", f"{res['mean_b']:.2f}",
           f"{res['diff']:.2f}", f"{res['se']:.2f}", f"{res['df']:.1f}", f"{res['t']:.3f}", f"{res['p']:.4g}",
           f"{res['diff'] - crit * res['se']:.2f}", f"{res['diff'] + crit * res['se']:.2f}",
           f'{p_bonf:.4g}', f"{res['diff'] - crit_bonf * res['se']:.2f}", f"{res['diff'] + crit_bonf * res['se']:.2f}",
           bool(res['p'] < ALPHA), bool(adjust and p_bonf < ALPHA)]
    for margin in MARGINS:
        p_tost = tost(res, margin)
        p_tost_bonf = min(1.0, M_TESTS * p_tost) if adjust else float('nan')
        out += [f'{p_tost:.4g}', f'{p_tost_bonf:.4g}', bool(p_tost < ALPHA), bool(adjust and p_tost_bonf < ALPHA)]
    return out


HEADER = ['comparison', 'env', 'window', 'h', 'rh', 'n_episodes_a', 'n_episodes_b', 'mean_a', 'mean_b', 'diff_a_minus_b',
          'se', 'welch_df', 'welch_t', 'p_welch', 'ci95_low', 'ci95_high', 'p_bonf', 'ci_bonf_low', 'ci_bonf_high',
          'sig_uncorrected', 'sig_bonf'] + [f'{name}{m}' for m in MARGINS for name in
                                           ('tost_p_pm', 'tost_p_bonf_pm', 'equiv_pm', 'equiv_bonf_pm')]


def percentile_ci(samples):
    return np.percentile(samples, [2.5, 97.5])


def cell_matrix(cells, keys):
    """(cell, seed) array over the seeds shared by all `keys` cells, and the seed list."""
    seeds = sorted(set.intersection(*[set(cells[k]) for k in keys]))
    return np.array([[cells[k][s] for s in seeds] for k in keys]), seeds


def best_cell_stats(cells_a, cells_b, rng, paired):
    """Best cell of arm a vs best cell of arm b: observed maxima, and bootstrap CI of max(a) - max(b).

    paired=True resamples the shared seeds once for both arms; paired=False resamples each arm on its own seeds.
    Cells with fewer than 2 seeds in either arm are skipped. Returns None when no cell qualifies.
    """
    keys = [k for k in cells_a if k in cells_b and len(cells_a[k]) > 1 and len(cells_b[k]) > 1]
    if not keys:
        return None
    if paired:
        seeds = sorted(set.intersection(*[set(cells[k]) for k in keys for cells in (cells_a, cells_b)]))
        a, b = (np.array([[cells[k][s] for s in seeds] for k in keys]) for cells in (cells_a, cells_b))
        draws_a = draws_b = rng.integers(0, len(seeds), (BOOTSTRAPS, len(seeds)))
    else:
        a, _ = cell_matrix(cells_a, keys)
        b, _ = cell_matrix(cells_b, keys)
        draws_a = rng.integers(0, a.shape[1], (BOOTSTRAPS, a.shape[1]))
        draws_b = rng.integers(0, b.shape[1], (BOOTSTRAPS, b.shape[1]))
    best_a = a[:, draws_a].mean(axis=2).max(axis=0)  # (cells, B, n) -> (B,)
    best_b = b[:, draws_b].mean(axis=2).max(axis=0)
    diff = best_a - best_b
    low, high = percentile_ci(diff)
    index_a, index_b = a.mean(axis=1).argmax(), b.mean(axis=1).argmax()
    return dict(cell_a=keys[index_a], mean_a=a.mean(axis=1).max(), cell_b=keys[index_b], mean_b=b.mean(axis=1).max(),
                diff=diff.mean(), lo=low, hi=high, star=low > 0 or high < 0, n_a=a.shape[1], n_b=b.shape[1])
