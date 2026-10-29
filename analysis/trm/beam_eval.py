"""TRM vs goal-MSE objective in the LeWAM planners (beam smoke protocol), per env.

Each planner (independent chains, top-k beam, soft beam τ=3, native gradient) runs once with goal-MSE and once with
the TRM head (eval.objective=trm, trm_scope=all), plus a top-k beam with the shuffled-label head as the control.
Reads results.txt (and episodes.csv for the TwoRoom same-room / cross-wall split); writes a table and one figure per
env to data/trm/analysis/beam_eval/. Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/trm/beam_eval.py
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "analysis" / "planner_comparison"))
from plot_style import INK_2, style  # noqa: E402

DATA = LW / "data"
SMOKE = DATA / "beam_smoke"
OUT = DATA / "trm" / "analysis" / "beam_eval"
SEEDS = [8902, 10748, 17401]
PLANNERS = ["chains", "topk", "soft_t3", "native"]
PLANNER_TEXT = {"chains": "independent chains\n(B32 C1)", "topk": "top-k beam\n(B8 C4)",
                "soft_t3": "soft beam τ=3\n(B8 C4)", "native": "native gradient"}
# goal-MSE runs: the existing smoke runs of each env (same protocol), new PointMaze baselines in trm/
MSE_DIRS = {
    "tworoom_hard": {"chains": SMOKE / "tworoom_hard/spread/chains_b32c1", "topk": SMOKE / "tworoom_hard/spread/topk_goal",
                     "soft_t3": SMOKE / "tworoom_hard/spread/soft_t3_goal", "native": DATA / "tworoom_hard/native_d50"},
    "puzzle": {"chains": SMOKE / "puzzle/best_of_k", "topk": SMOKE / "puzzle/trm/topk_goal",
               "soft_t3": SMOKE / "puzzle/beam_soft_t3", "native": SMOKE / "puzzle/native_grad"},
    "pointmaze": {"chains": SMOKE / "pointmaze/trm/chains_b32c1", "topk": SMOKE / "pointmaze/trm/topk_goal",
                  "soft_t3": SMOKE / "pointmaze/trm/soft_t3_goal", "native": SMOKE / "pointmaze/trm/native_grad"},
}
TRM_ARMS = {"chains": "chains_b32c1_trm", "topk": "topk_trm", "soft_t3": "soft_t3_trm", "native": "native_trm"}
COLORS = {"goal-MSE": "#2a78d6", "TRM": "#1b8a3a", "TRM shuffled": "#9a9890"}   # LeWAM blue, TRM green, control grey


def read_run(run_dir):
    """Success rate of one run, plus the cross-wall success rate when episodes.csv has the TwoRoom split."""
    match = re.search(r"'success_rate': ([0-9.]+)", (run_dir / "results.txt").read_text())
    row = dict(sr=float(match.group(1)))
    episodes = run_dir / "episodes.csv"
    if episodes.exists():
        frame = pd.read_csv(episodes)
        if "same_room" in frame:
            row["same_room"] = 100 * frame[frame.same_room].success.mean()
            row["cross_wall"] = 100 * frame[~frame.same_room].success.mean()
    return row


def collect():
    rows = []
    for env, mse_dirs in MSE_DIRS.items():
        for planner in PLANNERS:
            arms = {"goal-MSE": mse_dirs[planner], "TRM": SMOKE / env / "trm" / TRM_ARMS[planner]}
            if planner == "topk":
                arms["TRM shuffled"] = SMOKE / env / "trm" / "topk_trm_shuff"
            for objective, arm_dir in arms.items():
                for seed in SEEDS:
                    if (arm_dir / str(seed) / "results.txt").exists():
                        rows.append(dict(env=env, planner=planner, objective=objective, seed=seed,
                                         run=str(arm_dir.relative_to(DATA)), **read_run(arm_dir / str(seed))))
    return pd.DataFrame(rows)


def summarise(runs):
    metrics = [m for m in ["sr", "same_room", "cross_wall"] if m in runs]
    grouped = runs.groupby(["env", "planner", "objective"])[metrics]
    summary = grouped.mean().join(grouped.std(ddof=1).add_suffix("_sd")).join(grouped.size().rename("seeds"))
    return summary.reset_index()


def plot_env(env, summary, metrics, suffix):
    """Grouped bars, one panel per metric: per planner, goal-MSE vs TRM (vs shuffled TRM for top-k)."""
    part = summary[summary.env == env]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.5 * len(metrics), 4.6), squeeze=False)
    width = 0.27
    for ax, (metric, label) in zip(axes[0], metrics):
        for position, planner in enumerate(PLANNERS):
            bars = part[part.planner == planner].set_index("objective")
            objectives = [o for o in COLORS if o in bars.index]
            for index, objective in enumerate(objectives):
                x = position + (index - (len(objectives) - 1) / 2) * width
                value, sd = bars.loc[objective, metric], bars.loc[objective, f"{metric}_sd"]
                error = 0.0 if np.isnan(sd) else 2 * sd           # error bars are always ±2 SD
                # a success rate lives in [0, 100], so the bar is clipped there
                low, high = min(error, value), min(error, 100 - value)
                ax.bar(x, value, width * 0.92, color=COLORS[objective], yerr=[[low], [high]], capsize=2,
                       error_kw=dict(elinewidth=0.8, ecolor=INK_2))
                ax.text(x, 3, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5, color="#fcfcfb",
                        fontweight="bold")
        ax.set_xticks(range(len(PLANNERS)), [PLANNER_TEXT[p] for p in PLANNERS], fontsize=8.5)
        ax.set(ylim=(0, 105), ylabel=label)
    handles = [Patch(color=color) for color in COLORS.values()]
    fig.legend(handles, list(COLORS), loc="lower center", ncol=3, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle(f"{env}: goal-MSE vs TRM objective (trm_scope=all), mean of 3 seeds × 30 episodes, "
                 "error bars ±2 SD over seeds (clipped to [0, 100])", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / f"{env}_trm_vs_mse{suffix}.png", dpi=150)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect()
    runs.to_csv(OUT / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(summary.to_string(index=False))
    merged = [("sr", "success rate (%)")]
    split = [("same_room", "same-room success rate (%)"), ("cross_wall", "cross-wall success rate (%)")]
    for env in MSE_DIRS:
        plot_env(env, summary, merged, "")                       # all episodes
        if env == "tworoom_hard":
            plot_env(env, summary, split, "_split")              # same-room and cross-wall episodes apart
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
