"""Success rate of the goal-MSE beam arms (L6-L8, plus the goal-MSE runs of the PointMaze TRM eval), env x arm.

Reads results.txt (and episodes.csv for the TwoRoom cross-wall split) of each run; writes runs.csv, summary.csv, a
heatmap, a LaTeX table (sr_table.tex, booktabs) and a PNG preview of that table to data/beam_smoke/analysis/sr/.
Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/beam_tree/sr_table.py
"""

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
from plot_style import INK, INK_2, MUTED, SEQUENTIAL, style  # noqa: E402

DATA = LW / "data"
SMOKE = DATA / "beam_smoke"
OUT = SMOKE / "analysis" / "sr"
SEEDS = [8902, 10748, 17401]

ARMS = ["native", "chains", "soft_t10", "soft_t3", "soft_t1", "soft_t0p3", "soft_t0p1", "topk"]
ARM_TEXT = {
    "native": "native gradient (no beam)",
    "chains": "independent chains (best_of_k, B32 C1)",
    "soft_t10": "soft beam τ=10", "soft_t3": "soft beam τ=3", "soft_t1": "soft beam τ=1",
    "soft_t0p3": "soft beam τ=0.3", "soft_t0p1": "soft beam τ=0.1",
    "topk": "top-k beam",
}
# table preview labels in mathtext: cmr10 has no glyphs for τ, ±, ×, dashes or underscore outside math mode
ARM_MATH = {
    "native": r"native gradient (no beam)",
    "chains": r"independent chains ($\mathtt{best\_of\_k}$, B32 C1)",
    "soft_t10": r"soft beam $\tau=10$", "soft_t3": r"soft beam $\tau=3$", "soft_t1": r"soft beam $\tau=1$",
    "soft_t0p3": r"soft beam $\tau=0.3$", "soft_t0p1": r"soft beam $\tau=0.1$",
    "topk": r"top-k beam",
}
ARM_TEX = {
    "native": r"native gradient (no beam)",
    "chains": r"independent chains (\texttt{best\_of\_k}, B32 C1)",
    "soft_t10": r"soft beam $\tau=10$", "soft_t3": r"soft beam $\tau=3$", "soft_t1": r"soft beam $\tau=1$",
    "soft_t0p3": r"soft beam $\tau=0.3$", "soft_t0p1": r"soft beam $\tau=0.1$",
    "topk": r"top-k beam",
}
# TwoRoom hard: L7 batch (the spread reruns of chains / top-k / soft τ=1, 3 give the same means); native from L5
RUN_DIRS = {
    "tworoom_hard": {
        "native": DATA / "tworoom_hard/native_d50", "chains": SMOKE / "tworoom_hard/best_of_k",
        "soft_t10": SMOKE / "tworoom_hard/beam_soft_t10", "soft_t3": SMOKE / "tworoom_hard/beam_soft_t3",
        "soft_t1": SMOKE / "tworoom_hard/beam_soft_t1", "soft_t0p3": SMOKE / "tworoom_hard/beam_soft_t0p3",
        "soft_t0p1": SMOKE / "tworoom_hard/beam_soft_t0p1", "topk": SMOKE / "tworoom_hard/beam_goal",
    },
    "pusht": {
        "native": DATA / "planner_comparison/p1/runs/pusht/native/h5_rh1", "chains": SMOKE / "pusht/best_of_k",
        "soft_t10": SMOKE / "pusht/beam_soft_t10", "soft_t3": SMOKE / "pusht/beam_soft_t3",
        "soft_t1": SMOKE / "pusht/beam_soft_t1",
    },
    "puzzle": {
        "native": SMOKE / "puzzle/native_grad", "chains": SMOKE / "puzzle/best_of_k",
        "soft_t10": SMOKE / "puzzle/beam_soft_t10", "soft_t3": SMOKE / "puzzle/beam_soft_t3",
        "soft_t1": SMOKE / "puzzle/beam_soft_t1", "topk": SMOKE / "puzzle/trm/topk_goal",
    },
    "reacher": {
        "native": SMOKE / "reacher/native_grad", "chains": SMOKE / "reacher/best_of_k",
        "soft_t10": SMOKE / "reacher/beam_soft_t10", "soft_t3": SMOKE / "reacher/beam_soft_t3",
        "soft_t1": SMOKE / "reacher/beam_soft_t1",
    },
    "pointmaze": {
        "native": SMOKE / "pointmaze/trm/native_grad", "chains": SMOKE / "pointmaze/trm/chains_b32c1",
        "soft_t3": SMOKE / "pointmaze/trm/soft_t3_goal", "topk": SMOKE / "pointmaze/trm/topk_goal",
    },
}
# heatmap / table columns: (env, metric, label)
COLUMNS = [
    ("tworoom_hard", "sr", "TwoRoom hard"), ("tworoom_hard", "cross_wall", "TwoRoom hard\ncross-wall"),
    ("pusht", "sr", "PushT"), ("puzzle", "sr", "Puzzle"), ("reacher", "sr", "Reacher"),
    ("pointmaze", "sr", "PointMaze"),
]
CAPTION = ("success rate (%), goal-MSE pruning and final pick, mean of 3 seeds × 30 episodes, ± 2 SD over seeds; "
           "best per column in bold; — = not run\n"
           "TwoRoom hard: deadline 50, h10 rh1; PushT / Puzzle / Reacher: offset 25, h5 rh1; PointMaze: offset 50, "
           "h10; beams B8 C4, no gradient")
CAPTION_MATH = (r"success rate (%), goal-MSE pruning and final pick, mean of 3 seeds $\times$ 30 episodes, $\pm$ 2 SD "
                r"over seeds; best per column in bold; $-$ = not run" "\n"
                r"TwoRoom hard: deadline 50, h10 rh1; PushT / Puzzle / Reacher: offset 25, h5 rh1; PointMaze: "
                r"offset 50, h10; beams B8 C4, no gradient")
LATEX_FONT = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
              "axes.formatter.use_mathtext": True, "axes.unicode_minus": False}
MISSING = r"$-$"


def read_run(run_dir):
    """Success rate of one run, plus the cross-wall success rate when episodes.csv has the TwoRoom split."""
    match = re.search(r"'success_rate': ([0-9.]+)", (run_dir / "results.txt").read_text())
    row = dict(sr=float(match.group(1)))
    episodes = run_dir / "episodes.csv"
    if episodes.exists():
        frame = pd.read_csv(episodes)
        if "same_room" in frame:
            row["cross_wall"] = 100 * frame[~frame.same_room].success.mean()
    return row


def collect():
    rows = []
    for env, arm_dirs in RUN_DIRS.items():
        for arm, arm_dir in arm_dirs.items():
            for seed in SEEDS:
                run_dir = arm_dir / str(seed)
                if (run_dir / "results.txt").exists():
                    rows.append(dict(env=env, arm=arm, seed=seed, run=str(run_dir.relative_to(DATA)),
                                     **read_run(run_dir)))
    return pd.DataFrame(rows)


def summarise(runs):
    grouped = runs.groupby(["env", "arm"])[["sr", "cross_wall"]]
    summary = grouped.mean().join(grouped.std(ddof=1).add_suffix("_sd")).join(grouped.size().rename("seeds"))
    return summary.reset_index()


def grid(summary):
    """(mean, sd) arrays of shape (arms, columns); NaN where the arm was not run."""
    mean = np.full((len(ARMS), len(COLUMNS)), np.nan)
    sd = np.full_like(mean, np.nan)
    indexed = summary.set_index(["env", "arm"])
    for i, arm in enumerate(ARMS):
        for j, (env, metric, _) in enumerate(COLUMNS):
            if (env, arm) in indexed.index:
                mean[i, j] = indexed.loc[(env, arm), metric]
                sd[i, j] = indexed.loc[(env, arm), f"{metric}_sd"]
    return mean, sd


def best_mask(mean):
    """True where a cell ties the best mean of its column."""
    column_best = np.nanmax(mean, axis=0)
    return np.isclose(mean, column_best[None, :])


def plot_heatmap(mean, sd):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    shown = np.ma.masked_invalid(mean)
    cmap = SEQUENTIAL.with_extremes(bad="#efeee9")
    image = ax.imshow(shown, cmap=cmap, vmin=30, vmax=100, aspect="auto")
    best = best_mask(mean)
    for i in range(len(ARMS)):
        for j in range(len(COLUMNS)):
            if np.isnan(mean[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=11)
                continue
            text_colour = "#fcfcfb" if mean[i, j] > 70 else INK
            weight = "bold" if best[i, j] else "normal"
            ax.text(j, i, f"{mean[i, j]:.1f}\n±{2 * sd[i, j]:.1f}", ha="center", va="center",
                    color=text_colour, fontsize=9, fontweight=weight, linespacing=1.3)
    ax.set_xticks(range(len(COLUMNS)), [label for _, _, label in COLUMNS], fontsize=9)
    ax.set_yticks(range(len(ARMS)), [ARM_TEXT[arm] for arm in ARMS], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("success rate (%)", color=INK_2)
    colorbar.outline.set_visible(False)
    fig.text(0.5, 0.02, CAPTION, ha="center", va="bottom", fontsize=8, color=INK_2, wrap=True)
    fig.suptitle("LeWAM beam search with goal-MSE scoring: success rate per env and arm", fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "sr_heatmap.png", dpi=150)
    plt.close(fig)


def cell_text(mean, sd, best, tex):
    """One table cell as LaTeX math (also valid mathtext for the PNG preview)."""
    if np.isnan(mean):
        return "--" if tex else MISSING
    value = rf"\mathbf{{{mean:.1f}}}" if best else f"{mean:.1f}"
    return rf"${value} \pm {2 * sd:.1f}$"


def write_tex(mean, sd):
    best = best_mask(mean)
    header = " & ".join(["arm"] + [label.replace("\n", " ") for _, _, label in COLUMNS])
    body = []
    for i, arm in enumerate(ARMS):
        cells = [cell_text(mean[i, j], sd[i, j], best[i, j], tex=True) for j in range(len(COLUMNS))]
        body.append(" & ".join([ARM_TEX[arm]] + cells) + r" \\")
        if arm in ("chains", "soft_t0p1"):
            body.append(r"\midrule")
    lines = [
        r"\documentclass[border=4pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        r"% " + CAPTION.replace("\n", "\n% "),
        r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{document}",
    ]
    (OUT / "sr_table.tex").write_text("\n".join(lines) + "\n")


def plot_table(mean, sd):
    """PNG preview of sr_table.tex in booktabs style (no TeX engine on the server)."""
    best = best_mask(mean)
    labels = [ARM_MATH[arm] for arm in ARMS]
    n_rows, n_cols = len(ARMS), len(COLUMNS)
    widths = [3.1] + [1.45] * n_cols
    x_edges = np.concatenate([[0], np.cumsum(widths)])
    row_height = 0.36
    fig, ax = plt.subplots(figsize=(x_edges[-1] + 0.4, row_height * (n_rows + 2.6) + 0.9))
    ax.set_xlim(0, x_edges[-1])
    ax.set_ylim(-(n_rows + 1.6) * row_height, 0.25)
    ax.axis("off")

    def rule(y, width):
        ax.plot([0, x_edges[-1]], [y, y], color=INK, linewidth=width, solid_capstyle="butt")

    rule(0, 1.2)
    header_y = -0.75 * row_height
    ax.text(0.05, header_y, "arm", ha="left", va="center", fontsize=10.5)
    for j, (_, _, label) in enumerate(COLUMNS):
        centre = (x_edges[j + 1] + x_edges[j + 2]) / 2
        ax.text(centre, header_y, label, ha="center", va="center", fontsize=10.5, linespacing=1.1)
    y = -1.5 * row_height
    rule(y, 0.7)
    for i, arm in enumerate(ARMS):
        row_y = y - (i + 0.5) * row_height
        ax.text(0.05, row_y, labels[i], ha="left", va="center", fontsize=10.5)
        for j in range(n_cols):
            centre = (x_edges[j + 1] + x_edges[j + 2]) / 2
            text = cell_text(mean[i, j], sd[i, j], best[i, j], tex=False)
            colour = MUTED if np.isnan(mean[i, j]) else INK
            ax.text(centre, row_y, text, ha="center", va="center", fontsize=10.5, color=colour)
        if arm in ("chains", "soft_t0p1"):
            rule(y - (i + 1) * row_height, 0.5)
    rule(y - n_rows * row_height, 1.2)
    fig.text(0.5, 0.01, CAPTION_MATH, ha="center", va="bottom", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "sr_table.png", dpi=170)
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
    mean, sd = grid(summary)
    plot_heatmap(mean, sd)
    write_tex(mean, sd)
    with plt.rc_context(LATEX_FONT):
        plot_table(mean, sd)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
