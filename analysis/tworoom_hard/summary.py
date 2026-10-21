"""Summarise the LeWAM TwoRoom hard runs (scripts/tworoom_hard/run.sh) next to the stable-worldmodel CEM runs on the
same seeds and episodes (cem_vs_icem_seeds25, LeWM + CEM 300×10). Mean and per-seed success only, no tests.

Writes data/tworoom_hard/summary.md. Usage: python analysis/tworoom_hard/summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

LW = Path(__file__).resolve().parents[2]
DATA = LW / 'data' / 'tworoom_hard'
PG_RUNS = LW.parent / 'stable-worldmodel' / 'data' / 'experiment' / 'cem_vs_icem_seeds25' / 'runs.csv'
ARMS = {'native_d25': 'LeWAM native, deadline 25 (h0 = 5)', 'native_d50': 'LeWAM native, deadline 50 (h0 = 10)',
        'constant': 'LeWAM constant (5 blocks ahead)'}
SEEDS = [8902, 10748, 17401]


def load_lewam():
    """One row per (arm, seed, episode) from episodes.csv."""
    frames = []
    for arm in ARMS:
        for seed in SEEDS:
            path = DATA / arm / str(seed) / 'episodes.csv'
            if path.exists():
                frames.append(pd.read_csv(path).assign(arm=arm, seed=seed))
    return pd.concat(frames, ignore_index=True)


def load_pg_cem():
    """CEM success per (window, objective, h, rh, seed) on TwoRoom hard.

    Fields by position (header: date, exp, env, policy, solver, objective, h, rh, seed, sr, time); TwoRoom rows
    carry two unnamed extra fields, same-room and cross-wall success.
    """
    rows = []
    for line in PG_RUNS.read_text().splitlines()[1:]:
        fields = line.split(',')
        if '/tworoom_hard/cem_w' not in fields[1]:
            continue
        extra = fields[11:13] if len(fields) >= 13 else [np.nan, np.nan]
        rows.append(dict(window=fields[1].rsplit('_', 1)[-1], objective=fields[5], h=int(fields[6]),
                         rh=int(fields[7]), seed=int(fields[8]), sr=float(fields[9]),
                         same_room=float(extra[0]), cross_wall=float(extra[1])))
    cem = pd.DataFrame(rows)
    return cem[cem.seed.isin(SEEDS)]


def pct(values):
    return 100 * float(np.mean(values))


def lewam_table(episodes):
    lines = ['| Arm | success % (mean of seeds) | per seed | same-room % | cross-wall % | doorway-required % | '
             'final distance px, failures (median) | closest approach px, failures (median) |', '|' + '---|' * 8]
    for arm, label in ARMS.items():
        runs = episodes[episodes.arm == arm]
        if runs.empty:
            continue
        per_seed = runs.groupby('seed').success.mean() * 100
        failed = runs[~runs.success]
        lines.append(f"| {label} | {per_seed.mean():.1f} | {' / '.join(f'{v:.0f}' for v in per_seed)} | "
                     f"{pct(runs[runs.same_room].success):.1f} | {pct(runs[~runs.same_room].success):.1f} | "
                     f"{pct(runs[runs.requires_door].success):.1f} | "
                     f"{failed.final_distance.median():.0f} | {failed.min_distance.median():.0f} |")
    return lines


def cem_table(cem):
    lines = ['| CEM cell (LeWM, 300×10) | objective | success % (mean of seeds) | per seed | same-room % | '
             'cross-wall % |', '|---|---|---|---|---|---|']
    cells = cem.groupby(['window', 'objective', 'h', 'rh'])
    means = cells.sr.mean().sort_values(ascending=False)
    for key, mean in means.items():
        cell = cells.get_group(key).drop_duplicates('seed').set_index('seed')
        if len(cell) < len(SEEDS):
            continue  # cell not run on every seed
        window, objective, h, rh = key
        marker = ' (P2 setting)' if (h, rh) == (5, 1) else ''
        lines.append(f"| w={window[1:]} h{h} rh{rh}{marker} | {objective} | {mean:.1f} | "
                     f"{' / '.join(f'{v:.0f}' for v in cell.sr.reindex(SEEDS))} | "
                     f"{cell.same_room.mean():.1f} | {cell.cross_wall.mean():.1f} |")
    return lines


def main():
    episodes = load_lewam()
    cem = load_pg_cem()
    text = '\n'.join([
        '# LeWAM on TwoRoom hard, goal offset as a deadline (first look, 3 seeds × 30 episodes)', '',
        f'Seeds {SEEDS}; each seed\'s 30 episodes are the ones the CEM runs used (per-seed manifest, 15 same-room + '
        '15 cross-wall, budget 50 env steps). Success in percent; per-seed values in seed order. Final and closest '
        'distances are over failed episodes only (success radius 16 px). No statistical test.', '',
        *lewam_table(episodes), '',
        'CEM on the same seeds and episodes (stable-worldmodel cem_vs_icem_seeds25, LeWM model), cells run on all '
        'three seeds, best first:', '',
        *cem_table(cem), '',
    ])
    (DATA / 'summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
