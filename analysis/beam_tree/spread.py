"""Per-depth spread of the beam's 32 nodes (beam_spread.jsonl) and success rate, per env and arm.

Reads data/beam_smoke/<env>/spread/<arm>/<seed>/{beam_spread.jsonl,results.txt}; writes tables and one figure per
env to data/beam_smoke/analysis/spread/. Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/beam_tree/spread.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "analysis" / "planner_comparison"))
from plot_style import FAMILY_COLORS, INK_2, style  # noqa: E402

DATA = LW / "data" / "beam_smoke"
OUT = DATA / "analysis" / "spread"
METRICS = ["action_spread", "sibling_action_spread", "prefix_spread", "latent_spread", "distinct_roots", "goal_min"]
METRIC_LABELS = {
    "action_spread": "action spread (this depth's block)",
    "sibling_action_spread": "sibling action spread",
    "prefix_spread": "prefix spread (root → depth)",
    "latent_spread": "latent spread (this depth)",
    "distinct_roots": "distinct first moves",
    "goal_min": "best goal-MSE at depth",
}

# four hue groups, shades within: independent chains + top-k goal (LeWAM blue), soft goal (violet),
# top-k A* (magenta), soft A* (teal); A* arms drawn dashed with square markers
ARM_COLORS = {
    "chains_b32c1": FAMILY_COLORS["native"], "topk_goal": "#6ea8ee",
    "soft_t1_goal": "#4a2aa8", "soft_t3_goal": "#9b7fe3",
    "topk_astar_l1": "#a3245e", "topk_astar_l0p1": "#e07aa8",
    "soft_t3_astar_l1": "#0f6e78", "soft_t3_astar_l0p1": "#5bbdc6",
}
ARM_ORDER = ["chains_b32c1", "topk_goal", "soft_t1_goal", "soft_t3_goal",
             "topk_astar_l0p1", "topk_astar_l1", "soft_t3_astar_l0p1", "soft_t3_astar_l1"]


def arm_style(arm):
    """(colour, line style, marker, marker face) of an arm."""
    colour = ARM_COLORS[arm]
    if "_astar" in arm:
        return colour, "--", "s", colour
    return colour, "-", "o", colour


def load_spread():
    frames = []
    for path in sorted(DATA.glob("*/spread/*/*/beam_spread.jsonl")):
        env, _, arm, seed = path.parts[-5:-1]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        frame = pd.DataFrame(rows).assign(env=env, arm=arm, seed=int(seed))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_success():
    rows = []
    for path in sorted(DATA.glob("*/spread/*/*/results.txt")):
        env, _, arm, seed = path.parts[-5:-1]
        match = re.search(r"'success_rate': ([0-9.]+)", path.read_text())
        if match:
            rows.append(dict(env=env, arm=arm, seed=int(seed), success_rate=float(match.group(1))))
    return pd.DataFrame(rows)


PRUNE_ARMS = ["chains_b32c1", "topk_goal", "soft_t1_goal", "soft_t3_goal"]
ASTAR_ROWS = {"top-k": ["chains_b32c1", "topk_goal", "topk_astar_l0p1", "topk_astar_l1"],
              "soft τ=3": ["chains_b32c1", "soft_t3_goal", "soft_t3_astar_l0p1", "soft_t3_astar_l1"]}
ASTAR_METRICS = ["action_spread", "prefix_spread", "distinct_roots", "goal_min"]
SUBTITLE = "mean of the per-seed means, error bars ±1 SD over seeds; rows past the deadline left out"


def draw(ax, by_depth, arms, metric):
    """One panel: metric against depth, one line per arm present in by_depth."""
    for index, arm in enumerate(arms):
        curve = by_depth[by_depth.arm == arm].sort_values("depth")
        colour, line, marker, face = arm_style(arm)
        offset = (index - (len(arms) - 1) / 2) * 0.05             # spread the error bars of one depth apart
        ax.errorbar(curve.depth + offset, curve[metric], yerr=curve[f"{metric}_sd"], color=colour,
                    linestyle=line, marker=marker, markerfacecolor=face, markersize=4, linewidth=1.4,
                    elinewidth=0.8, capsize=2, label=arm)
    ax.set_xlabel("depth (blocks)", fontsize=9)


def legend(fig, ax, success_rates):
    handles, arms = ax.get_legend_handles_labels()
    labels = [f"{arm}  (SR {success_rates.get(arm, float('nan')):.1f})" for arm in arms]
    fig.legend(handles, labels, loc="lower center", ncol=len(arms), fontsize=8.5, labelcolor=INK_2)


def plot_prune(env, by_depth, success_rates):
    """Prune rule with the goal score: independent chains vs top-k vs soft τ=1 / τ=3, all six metrics."""
    arms = [arm for arm in PRUNE_ARMS if arm in set(by_depth.arm)]
    if len(arms) < 2:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, metric in zip(axes.flat, METRICS):
        draw(ax, by_depth, arms, metric)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
    legend(fig, axes.flat[0], success_rates)
    fig.suptitle(f"{env}, prune rule (goal score): spread of the 32 beam nodes per depth\n{SUBTITLE}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / f"{env}_spread_prune.png", dpi=150)
    plt.close(fig)


def plot_astar(env, by_depth, success_rates):
    """Goal score vs A* (λ 0.1, 1), one row per prune rule, independent chains as the reference."""
    rows = {name: [arm for arm in arms if arm in set(by_depth.arm)] for name, arms in ASTAR_ROWS.items()}
    rows = {name: arms for name, arms in rows.items() if any("_astar" in arm for arm in arms)}
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(ASTAR_METRICS), figsize=(14, 3.6 * len(rows) + 0.8), squeeze=False)
    for row_axes, (name, arms) in zip(axes, rows.items()):
        for ax, metric in zip(row_axes, ASTAR_METRICS):
            draw(ax, by_depth, arms, metric)
            ax.set_title(f"{name}: {METRIC_LABELS[metric]}", fontsize=9.5)
        legend_row = row_axes[0].get_legend_handles_labels()
        row_axes[-1].legend(legend_row[0], [f"{arm} ({success_rates.get(arm, float('nan')):.1f})"
                                            for arm in legend_row[1]], fontsize=7.5, labelcolor=INK_2)
    fig.suptitle(f"{env}, goal score vs A* within each prune rule (legend: arm, SR)\n{SUBTITLE}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"{env}_spread_astar.png", dpi=150)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    spread = load_spread()
    active = spread[~spread.frozen]
    # seeds are the independent unit: average within each seed first, then mean and SD over the seed means
    seed_means = active.groupby(["env", "arm", "seed", "depth"])[METRICS].mean().reset_index()
    grouped = seed_means.groupby(["env", "arm", "depth"])[METRICS]
    by_depth = grouped.mean().join(grouped.std(ddof=1).add_suffix("_sd")).reset_index()
    by_depth["seeds"] = grouped.size().values
    by_depth["rows"] = active.groupby(["env", "arm", "depth"]).size().values
    by_depth.to_csv(OUT / "spread_by_depth.csv", index=False)

    success = load_success()
    success.to_csv(OUT / "success_runs.csv", index=False)
    summary = success.groupby(["env", "arm"]).success_rate.agg(["mean", "count"]).reset_index()
    summary.to_csv(OUT / "success_summary.csv", index=False)
    print(summary.to_string(index=False))

    for env in sorted(by_depth.env.unique()):
        success_rates = success[success.env == env].groupby("arm").success_rate.mean()
        plot_prune(env, by_depth[by_depth.env == env], success_rates)
        plot_astar(env, by_depth[by_depth.env == env], success_rates)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
