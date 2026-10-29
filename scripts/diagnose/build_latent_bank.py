"""Encode a latent bank of dataset frames under one LeWAM checkpoint.

  python scripts/diagnose/build_latent_bank.py --policy lewam-pusht --dataset stackt_test --n-frames 20000 \
      --out data/beam_smoke/latent_banks/lewam-pusht__stackt_test.pt
  (--half 0 / 1: draw from one of two disjoint episode halves, for a training set's two banks)
"""

import argparse
import time
from pathlib import Path

import torch
from stable_worldmodel.data.utils import get_cache_dir

from lewam.diagnose.references import build_bank
from lewam.eval.loaders import load_lewam


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-frames", type=int, default=20000)
    parser.add_argument("--half", type=int, choices=(0, 1), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    assert not out.exists(), f"{out} exists: never overwrite a bank"
    out.parent.mkdir(parents=True, exist_ok=True)
    h5_path = Path(get_cache_dir(sub_folder="datasets")) / f"{args.dataset}.h5"

    model, _ = load_lewam(args.policy)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval().requires_grad_(False)
    start = time.time()
    bank = build_bank(model, h5_path, args.policy, args.n_frames, seed=args.seed, half=args.half)
    bank.save(out)
    print(f"wrote {out}: {len(bank.latents)} frames from {len(bank.episodes.unique())} episodes of {bank.dataset} "
          f"(half {bank.half}), z {bank.latents.shape[1]}, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
