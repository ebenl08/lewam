"""Does the TRM head still rank on LeWAM-imagined latents? Spearman vs depth k, real vs imagined vs goal-MSE.

On the head's held-out episodes, per depth k = 1..10 blocks: draw balanced-Delta pairs (position t + 5k, goal t_g),
roll LeWAM dynamics k blocks from the real history [t - 5, t] with the dataset's actions, and rank against the steps
left Delta = t_g - (t + 5k):
- rho_real: m(z_{t+5k}, z_g), the encoded real frame (the head without drift)
- rho_imag: m(zhat_{t+5k}, z_g), the imagined latent (what the planner scores)
- rho_mse:  goal-MSE of zhat_{t+5k} (today's objective)

    python -m lewam.trm.probe_imagined --dataset tworoom.h5 --checkpoint lewam-tworoom \\
        --head data/trm/heads/trm_tworoom_v1/weights_epoch_20.pt
"""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  the swm h5 files compress frames with plugin filters
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from stable_worldmodel.data.utils import get_cache_dir

from lewam.eval.loaders import load_lewam
from lewam.trm.cache import default_cache_path, load_cache
from lewam.trm.checkpoint import load_head
from lewam.trm.pairs import build_segments, sample_pairs

LW_ROOT = Path(__file__).resolve().parents[2]
COLOURS = {"rho_real": "#1b5e20", "rho_imag": "#66bb6a", "rho_mse": "#e65100"}
LABELS = {"rho_real": "TRM, real latent", "rho_imag": "TRM, imagined latent", "rho_mse": "goal-MSE, imagined latent"}


def probe_pairs(cache, val_seg, k, n_pairs, rng, frameskip, history_stride, sampler_cfg):
    """(t, i, j, delta) for depth k: i = t + k * frameskip is the scored position, j the goal, inside one val episode."""
    lengths, offsets = build_segments(cache.episode_idx)
    lengths, offsets = lengths[val_seg], offsets[val_seg]
    i, j, delta = sample_pairs(lengths, offsets, 4 * n_pairs, rng, mode=sampler_cfg.mode, n_bins=sampler_cfg.n_bins,
                               min_delta=sampler_cfg.min_delta, max_delta=sampler_cfg.max_delta, flip=False)
    t = i - k * frameskip
    episode_start = offsets[np.searchsorted(offsets, i, side="right") - 1]
    keep = t - history_stride >= episode_start            # whole history inside the episode, no padding
    keep_idx = np.flatnonzero(keep)[:n_pairs]
    return t[keep_idx], i[keep_idx], j[keep_idx], delta[keep_idx]


@torch.no_grad()
def imagine(model, cache, actions, t, k, frameskip, history_stride, device):
    """(B, z) latent after k imagined blocks from the real history [t - stride, t], driven by dataset actions."""
    history = torch.from_numpy(cache.emb[np.stack([t - history_stride, t], 1)]).to(device)   # (B, 2, z)
    history_pad = torch.zeros(len(t), 2, dtype=torch.bool, device=device)
    for block in range(k):
        rows = (t + block * frameskip)[:, None] + np.arange(frameskip)
        block_actions = torch.from_numpy(actions[rows]).to(device)                               # (B, fs, a)
        z_next = model.predict_states(history, history_pad, block_actions)[:, :1]                # (B, 1, z)
        history = torch.cat([history[:, 1:], z_next], 1)
    return history[:, -1]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head", required=True, help="head weights; its run dir holds val_segments.npy and config.yaml")
    p.add_argument("--n-pairs", type=int, default=4000)
    p.add_argument("--max-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: LW/data/trm/probe/<head run>/")
    args = p.parse_args(argv)

    head_path = Path(args.head)
    run_dir = head_path.parent
    train_cfg = OmegaConf.load(run_dir / "config.yaml")
    val_seg = np.load(run_dir / "val_segments.npy")
    out = Path(args.out) if args.out else LW_ROOT / "data" / "trm" / "probe" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_lewam(args.checkpoint)
    model = model.to(device).eval().requires_grad_(False)
    head = load_head(head_path).to(device).eval().requires_grad_(False)
    frameskip, history_stride = int(model_cfg["fs"]), int(model_cfg["history_stride"])

    cache_path = train_cfg.latent_cache or default_cache_path(args.dataset, args.checkpoint)
    cache = load_cache(cache_path, expect_checkpoint=args.checkpoint, expect_dataset=args.dataset)
    with h5py.File(Path(get_cache_dir(sub_folder="datasets")) / args.dataset, "r") as h5:
        ep_offset, ep_len = h5["ep_offset"][:], h5["ep_len"][:]
        raw = np.concatenate([h5["action"][int(ep_offset[e]):int(ep_offset[e] + ep_len[e])]
                              for e in cache.meta["episode_ids"]])                               # cache row order
    actions = ((raw - np.asarray(model_cfg["action_mean"])) / np.asarray(model_cfg["action_std"])).astype(np.float32)
    actions = np.nan_to_num(actions)                       # last action of an episode can be NaN; never imagined past

    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(1, args.max_k + 1):
        t, i, j, delta = probe_pairs(cache, val_seg, k, args.n_pairs, rng, frameskip, history_stride, train_cfg.sampler)
        z_goal = torch.from_numpy(cache.emb[j]).to(device)
        z_real = torch.from_numpy(cache.emb[i]).to(device)
        z_imag = imagine(model, cache, actions, t, k, frameskip, history_stride, device)
        scores = dict(
            rho_real=head(z_real, z_goal),
            rho_imag=head(z_imag, z_goal),
            rho_mse=((z_imag - z_goal) ** 2).mean(-1),
        )
        row = dict(k=k, n=len(t), delta_min=int(delta.min()), delta_max=int(delta.max()),
                   **{name: float(spearmanr(s.cpu().numpy(), delta).statistic) for name, s in scores.items()})
        rows.append(row)
        logging.info("  ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                               for key, value in row.items()))

    header = list(rows[0])
    (out / "probe.csv").write_text("\n".join([",".join(header)] + [",".join(str(r[h]) for h in header) for r in rows]))
    (out / "probe_meta.json").write_text(json.dumps(vars(args), indent=2))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ks = [r["k"] for r in rows]
    for name, colour in COLOURS.items():
        ax.plot(ks, [r[name] for r in rows], marker="o", color=colour, label=LABELS[name])
    ax.set(xlabel="imagined depth k (blocks)", ylabel="Spearman with steps left", title=run_dir.name, ylim=(-0.1, 1.0))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "probe.png", dpi=150)
    logging.info(f"probe written to {out}")


if __name__ == "__main__":
    main()
