"""Job list for the LeWAM planner comparison (notes/pending_implementation/lewam-planner-comparison/).

Prints one tab-separated job per missing run: phase, env, arm, h, rh, seed. A run is skipped when its output folder
already holds a finished results.txt.
- repro (P1a): `native` paper cell (h = rh = 0 means the paper default: h0 blocks, rh 1), seeds 42 / 0 / 1.
- grid (P1b): 3 arms x 9 (h, rh) cells x the 25 baseline seeds, PushT.

CLI (from playground/lewam/):
    python scripts/planner_comparison/make_jobs.py --phase repro > logs/planner_comparison/jobs.tsv
    python scripts/planner_comparison/make_jobs.py --phase grid --summary
"""

import argparse
import collections
import random
from pathlib import Path

ROOT = Path('data') / 'planner_comparison'
REPRO_ENVS = ('pusht', 'tworoom', 'reacher')
REPRO_SEEDS = (42, 0, 1)
GRID_ENVS = ('pusht',)
ARMS = ('native', 'grid_w1', 'grid_wH')
CELLS = [(h, rh) for h in (2, 5, 10) for rh in (1, 2, 5, 10) if rh <= h]
# the 25 seeds of PG scripts/eval/cem_vs_icem_seeds25/make_jobs.py
SEEDS = sorted([3679, 12083, 13483, 14596, 25661,
                8902, 10748, 17401, 18714, 28967, 30531, 36136, 37182, 47259, 60813,
                64351, 65625, 70738, 72399, 79261, 83275, 85333, 90609, 92065, 96729])


def run_dir(phase, env, arm, h, rh, seed):
    """Output folder of one run (mirrored by run.sh)."""
    if phase == 'repro':
        return ROOT / 'p1' / 'repro' / env / str(seed)
    return ROOT / 'p1' / 'runs' / env / arm / f'h{h}_rh{rh}' / str(seed)


def finished(path):
    results = path / 'results.txt'
    return results.exists() and "'success_rate'" in results.read_text()


def build_jobs(phase):
    if phase == 'repro':
        candidates = [('repro', env, 'native', 0, 0, seed) for env in REPRO_ENVS for seed in REPRO_SEEDS]
    else:
        candidates = [('grid', env, arm, h, rh, seed)
                      for env in GRID_ENVS for arm in ARMS for h, rh in CELLS for seed in SEEDS]
    return [job for job in candidates if not finished(run_dir(*job))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('repro', 'grid'), required=True)
    parser.add_argument('--summary', action='store_true', help='print counts per (env, arm) instead')
    args = parser.parse_args()
    jobs = build_jobs(args.phase)
    if args.summary:
        for key, count in collections.Counter(job[1:3] for job in jobs).items():
            print(*key, count)
        print('total', len(jobs))
        return
    random.Random(0).shuffle(jobs)  # spread the long cells over the workers
    for job in jobs:
        print(*job, sep='\t')


if __name__ == '__main__':
    main()
