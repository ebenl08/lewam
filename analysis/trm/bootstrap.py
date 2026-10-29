"""TwoRoom hard, 10 seeds: goal-MSE vs TRM objective per planner, hierarchical-bootstrap success rate and 95% CI.

Bootstrap: resample seeds with replacement, then episodes with replacement inside each drawn seed; the success rate
of a resample is the mean over its episodes. TRM - MSE differences are paired: both arms of a planner ran the same
seeds and manifest episodes, so one draw of (seed, episode) indices is applied to both. Overall and cross-wall.
Writes data/trm/analysis/bootstrap/. Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/trm/bootstrap.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "analysis" / "planner_comparison"))
from plot_style import INK_2, MUTED, style  # noqa: E402

DATA = LW / "data"
SMOKE = DATA / "beam_smoke" / "tworoom_hard"
OUT = DATA / "trm" / "analysis" / "bootstrap"
SEEDS = [8902, 10748, 17401, 18714, 28967, 30531, 36136, 37182, 47259, 60813]
N_BOOT, RNG_SEED = 10_000, 0
PLANNERS = ["chains", "topk", "soft_t3", "soft_t10", "native"]
PLANNER_TEXT = {"chains": "independent chains (B32 C1)", "topk": "top-k beam (B8 C4)", "soft_t3": "soft beam τ=3",
                "soft_t10": "soft beam τ=10", "native": "native gradient"}
# every arm's seed folders, first match wins: the 3-seed runs in their original groups, the new seeds in trm/
ARM_DIRS = {
    ("chains", "goal-MSE"): [SMOKE / "spread/chains_b32c1", SMOKE / "trm/chains_b32c1"],
    ("topk", "goal-MSE"): [SMOKE / "spread/topk_goal", SMOKE / "trm/topk_goal"],
    ("soft_t3", "goal-MSE"): [SMOKE / "spread/soft_t3_goal", SMOKE / "trm/soft_t3_goal"],
    ("soft_t10", "goal-MSE"): [SMOKE / "beam_soft_t10", SMOKE / "trm/soft_t10_goal"],
    # grad_tr 0.0 like native_trm (the grid runs used 0.01, so they do not pair with it)
    ("native", "goal-MSE"): [SMOKE / "trm/native_grad"],
    ("chains", "TRM"): [SMOKE / "trm/chains_b32c1_trm"],
    ("topk", "TRM"): [SMOKE / "trm/topk_trm"],
    ("soft_t3", "TRM"): [SMOKE / "trm/soft_t3_trm"],
    ("soft_t10", "TRM"): [SMOKE / "trm/soft_t10_trm"],
    ("native", "TRM"): [SMOKE / "trm/native_trm"],
}
COLORS = {"goal-MSE": "#2a78d6", "TRM": "#1b8a3a"}


def load_arm(dirs):
    """(seeds, episodes) success matrix and the same_room mask, rows in SEEDS order, columns in manifest row order."""
    success, same_room = [], []
    for seed in SEEDS:
        path = next((d / str(seed) / "episodes.csv" for d in dirs if (d / str(seed) / "episodes.csv").exists()), None)
        if path is None:
            raise FileNotFoundError(f"no episodes.csv for seed {seed} in {[str(d) for d in dirs]}")
        frame = pd.read_csv(path).sort_values("row")
        success.append(frame.success.to_numpy(bool))
        same_room.append(frame.same_room.to_numpy(bool))
    return np.stack(success), np.stack(same_room)


def bootstrap_indices(n_seeds, n_episodes, rng):
    """(N_BOOT, n_seeds) seed draws and (N_BOOT, n_seeds, n_episodes) episode draws inside each drawn seed."""
    seed_draw = rng.integers(0, n_seeds, size=(N_BOOT, n_seeds))
    episode_draw = rng.integers(0, n_episodes, size=(N_BOOT, n_seeds, n_episodes))
    return seed_draw, episode_draw


def resampled_rates(success, seed_draw, episode_draw):
    """(N_BOOT,) success rate (%) of every bootstrap resample of an (n_seeds, n_episodes) success matrix."""
    picked = success[seed_draw[:, :, None], episode_draw]
    return 100 * picked.mean(axis=(1, 2))


def planner_axis(ax):
    """Planner names as ticks; vertical separators between planners instead of grid lines through them."""
    ax.set_xticks(range(len(PLANNERS)), [PLANNER_TEXT[p].replace(" (", "\n(") for p in PLANNERS], fontsize=8)
    ax.grid(axis="x", visible=False)
    for boundary in range(1, len(PLANNERS)):
        ax.axvline(boundary - 0.5, color="#e1e0d9", linewidth=0.8)
    ax.set_xlim(-0.5, len(PLANNERS) - 0.5)


def interval(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return low, high


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {key: load_arm(dirs) for key, dirs in ARM_DIRS.items()}
    reference_rooms = arms[("chains", "goal-MSE")][1]
    assert all(np.array_equal(rooms, reference_rooms) for _, rooms in arms.values()), "arms ran different episodes"
    # per-seed episode masks; every seed has the same number of cross-wall episodes, so subsets stay rectangular
    subsets = {"overall": np.ones_like(reference_rooms), "cross-wall": ~reference_rooms}

    def select(success, mask):
        return np.stack([row[keep] for row, keep in zip(success, mask)])

    rng = np.random.default_rng(RNG_SEED)
    rows, diffs = [], []
    for subset, mask in subsets.items():
        n_episodes = int(mask[0].sum())
        assert (mask.sum(1) == n_episodes).all(), f"{subset}: episode count differs across seeds"
        seed_draw, episode_draw = bootstrap_indices(len(SEEDS), n_episodes, rng)
        rates = {}
        for (planner, objective), (success, _) in arms.items():
            part = select(success, mask)
            rates[planner, objective] = resampled_rates(part, seed_draw, episode_draw)
            low, high = interval(rates[planner, objective])
            rows.append(dict(subset=subset, planner=planner, objective=objective, sr=100 * part.mean(),
                             boot_mean=rates[planner, objective].mean(), ci_low=low, ci_high=high,
                             episodes=part.size))
        for planner in PLANNERS:
            delta = rates[planner, "TRM"] - rates[planner, "goal-MSE"]
            low, high = interval(delta)
            point = 100 * (select(arms[planner, "TRM"][0], mask).mean()
                           - select(arms[planner, "goal-MSE"][0], mask).mean())
            diffs.append(dict(subset=subset, planner=planner, trm_minus_mse=point, ci_low=low, ci_high=high,
                              p_le_0=(delta <= 0).mean()))
    table, diff_table = pd.DataFrame(rows), pd.DataFrame(diffs)
    table.to_csv(OUT / "success_ci.csv", index=False)
    diff_table.to_csv(OUT / "trm_minus_mse_ci.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(table.to_string(index=False))
        print(diff_table.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    for ax, subset in zip(axes[:2], subsets):
        part = table[table.subset == subset]
        for position, planner in enumerate(PLANNERS):
            for offset, objective in ((-0.15, "goal-MSE"), (0.15, "TRM")):
                row = part[(part.planner == planner) & (part.objective == objective)].iloc[0]
                ax.errorbar(position + offset, row.sr, yerr=[[row.sr - row.ci_low], [row.ci_high - row.sr]], fmt="o",
                            color=COLORS[objective], markersize=7, capsize=3, linewidth=1.4)
                side = -1 if objective == "goal-MSE" else 1          # MSE label left of its point, TRM right
                ax.text(position + offset + side * 0.06, row.sr, f"{row.sr:.1f}", fontsize=7.5, color=INK_2,
                        va="center", ha="right" if side < 0 else "left")
        planner_axis(ax)
        ax.set(ylim=(30, 101), ylabel=f"{subset} success rate (%)")
        ax.set_title(f"{subset}: success rate, 95% bootstrap CI", fontsize=10)
    ax = axes[2]
    for subset, marker, offset in (("overall", "o", -0.12), ("cross-wall", "s", 0.12)):
        part = diff_table[diff_table.subset == subset]
        for position, planner in enumerate(PLANNERS):
            row = part[part.planner == planner].iloc[0]
            ax.errorbar(position + offset, row.trm_minus_mse,
                        yerr=[[row.trm_minus_mse - row.ci_low], [row.ci_high - row.trm_minus_mse]], fmt=marker,
                        color=INK_2, markersize=6, capsize=3, linewidth=1.2)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    planner_axis(ax)
    ax.set(ylabel="TRM − goal-MSE (points)")
    ax.set_title("paired difference, 95% bootstrap CI", fontsize=10)
    ax.legend([Line2D([], [], color=INK_2, marker="o", linestyle=""), Line2D([], [], color=INK_2, marker="s", linestyle="")],
              ["overall", "cross-wall"], fontsize=8, labelcolor=INK_2)
    fig.legend([Line2D([], [], color=c, marker="o", linestyle="") for c in COLORS.values()], list(COLORS),
               loc="lower center", ncol=2, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle(f"TwoRoom hard, {len(SEEDS)} seeds × 30 episodes (15 cross-wall); hierarchical bootstrap over seeds "
                 f"and episodes, {N_BOOT:,} resamples; TRM uses trm_scope=all", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "tworoom_hard_bootstrap.png", dpi=150)
    plt.close(fig)
    plot_bars(table, diff_table, list(subsets))
    print(f"wrote {OUT}")


def ci_error(row, value):
    return [[value - row.ci_low], [row.ci_high - value]]


def plot_bars(table, diff_table, subsets):
    """Bar version of the same figure: success rate per arm and the paired difference, bars from 0, 95% CI whiskers."""
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, subset in zip(axes[:2], subsets):
        part = table[table.subset == subset]
        for position, planner in enumerate(PLANNERS):
            for offset, objective in ((-width / 2, "goal-MSE"), (width / 2, "TRM")):
                row = part[(part.planner == planner) & (part.objective == objective)].iloc[0]
                ax.bar(position + offset, row.sr, width * 0.92, color=COLORS[objective],
                       yerr=ci_error(row, row.sr), capsize=2, error_kw=dict(elinewidth=0.8, ecolor=INK_2))
                ax.text(position + offset, 3, f"{row.sr:.1f}", ha="center", va="bottom", fontsize=7.5,
                        color="#fcfcfb", fontweight="bold")
        planner_axis(ax)
        ax.set(ylim=(0, 105), ylabel=f"{subset} success rate (%)")
        ax.set_title(f"{subset}: success rate, 95% bootstrap CI", fontsize=10)
    ax = axes[2]
    diff_colors = {"overall": "#52514e", "cross-wall": "#a3a29b"}
    for offset, subset in ((-width / 2, "overall"), (width / 2, "cross-wall")):
        part = diff_table[diff_table.subset == subset]
        for position, planner in enumerate(PLANNERS):
            row = part[part.planner == planner].iloc[0]
            ax.bar(position + offset, row.trm_minus_mse, width * 0.92, color=diff_colors[subset],
                   yerr=ci_error(row, row.trm_minus_mse), capsize=2, error_kw=dict(elinewidth=0.8, ecolor=INK_2))
    ax.axhline(0, color=INK_2, linewidth=0.8)
    planner_axis(ax)
    ax.set(ylabel="TRM − goal-MSE (points)")
    ax.set_title("paired difference, 95% bootstrap CI", fontsize=10)
    ax.legend([Patch(color=color) for color in diff_colors.values()], list(diff_colors), fontsize=8,
              labelcolor=INK_2)
    fig.legend([Patch(color=c) for c in COLORS.values()], list(COLORS), loc="lower center", ncol=2, fontsize=8.5,
               labelcolor=INK_2)
    fig.suptitle(f"TwoRoom hard, {len(SEEDS)} seeds × 30 episodes (15 cross-wall); hierarchical bootstrap over seeds "
                 f"and episodes, {N_BOOT:,} resamples; TRM uses trm_scope=all", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "tworoom_hard_bootstrap_bars.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
