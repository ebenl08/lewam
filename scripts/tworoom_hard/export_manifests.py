"""Export the TwoRoom hard episodes of the stable-worldmodel CEM runs (cem_vs_icem_seeds25) for the LeWAM eval.

For each seed: the manifest PG eval_tworoom_hard.py builds in memory (build_manifest(seed)) and its stratified
num_eval-row subset (select_manifest_subset), saved as <out>/seed<seed>_n<num_eval>.npz. PG is only imported.

Usage (swm env, from playground/stable-worldmodel):
  PYTHONPATH=src python ../lewam/scripts/tworoom_hard/export_manifests.py --seeds 8902 10748 17401
"""

import argparse
import json
from pathlib import Path

import numpy as np

from envs.tworoom.manifest import build_manifest, select_manifest_subset

LW = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--num-eval', type=int, default=30)
    parser.add_argument('--out', default=str(LW / 'data' / 'tworoom_hard' / 'manifests'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        built = build_manifest(seed=seed)
        rows = select_manifest_subset(built['same_room'], args.num_eval)
        arrays = {k: v[rows] for k, v in built.items() if not k.startswith('_')}
        path = out / f'seed{seed}_n{args.num_eval}.npz'
        np.savez(path, row=rows, **arrays)
        (out / f'seed{seed}_stats.json').write_text(json.dumps(built['_stats'], indent=1))
        print(f'{path}: {len(rows)} rows, {int(arrays["same_room"].sum())} same-room, '
              f'{int(arrays["requires_door"].sum())} doorway-required')


if __name__ == '__main__':
    main()
