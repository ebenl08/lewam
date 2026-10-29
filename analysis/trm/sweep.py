"""TRM head training curves (baseline wandb reruns) and the Δ ≤ 50 hyperparameter sweep, per env.

Reads data/trm/heads_wandb/<run>/metrics.json and data/trm/sweep/<env>/<run>/{metrics.json,config.yaml}; writes
figures and tables to data/trm/analysis/. Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/trm/sweep.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "analysis" / "planner_comparison"))
from plot_style import INK_2, MUTED, style  # noqa: E402

DATA = LW / "data" / "trm"
OUT = DATA / "analysis"
ENVS = ["tworoom", "puzzle", "pointmaze"]
# one hue (TRM green, as in the probe plots), darker = tighter Δ cap; shuffled controls dashed
CAP_COLORS = {"d50": "#1b5e20", "d100": "#43a047", "full": "#a5d6a7"}
# sweep: categorical hues (validated, CVD-safe on the light surface): colour = lr, marker = n_pairs;
# the top-5 curves take slots 1..5 and the default config is grey dashed
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LR_COLORS = {3e-4: CATEGORICAL[0], 1e-3: CATEGORICAL[1], 3e-3: CATEGORICAL[2]}
PAIR_MARKERS = {100: "o", 1000: "^"}
DEFAULT_RUN = "lr1e-3_np100k_wd1e-4_do0.0_ep20"
HPARAMS = {"lr": "optimizer.lr", "n_pairs_k": "sampler.n_pairs", "weight_decay": "optimizer.weight_decay",
           "dropout": "head.dropout", "epochs": "trainer.max_epochs"}
HPARAM_TEXT = {"lr": lambda v: f"lr {v:g}", "n_pairs_k": lambda v: f"{v}k pairs", "weight_decay": lambda v: f"wd {v:g}",
               "dropout": lambda v: f"dropout {v:g}", "epochs": lambda v: f"{v} ep"}


def describe(row, reference=None):
    """Full config text, or only the settings that differ from `reference`."""
    names = [name for name in HPARAMS if reference is None or row[name] != reference[name]]
    return ", ".join(HPARAM_TEXT[name](row[name]) for name in names) or "same as #1"


def label_std(log_path, split="val"):
    """Std of the split's Δ labels, from the trainer's label_stats line."""
    match = re.search(rf"{split} labels: .* std=([0-9.]+)", log_path.read_text())
    return float(match.group(1))


def load_baseline(env):
    rows = []
    for run_dir in sorted((DATA / "heads_wandb").glob(f"trm_{env}_v1*")):
        if not (run_dir / "metrics.json").exists():          # still training
            continue
        history = pd.DataFrame(json.loads((run_dir / "metrics.json").read_text()))
        cap = re.search(r"_d(\d+)", run_dir.name)
        log = LW / "logs" / "trm" / env / "baseline_wandb" / f"{run_dir.name}.log"
        history["rmse_over_std"] = history["val/rmse_frames"] / label_std(log)
        rows.append(history.assign(run=run_dir.name, cap=f"d{cap.group(1)}" if cap else "full",
                                   shuffled=run_dir.name.endswith("_shuff")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_sweep(env):
    rows, curves = [], []
    for run_dir in sorted((DATA / "sweep" / env).glob("*/metrics.json")):
        run_dir = run_dir.parent
        cfg = OmegaConf.load(run_dir / "config.yaml")
        history = json.loads((run_dir / "metrics.json").read_text())
        row = {name: OmegaConf.select(cfg, key) for name, key in HPARAMS.items()}
        row["n_pairs_k"] //= 1000
        row.update(run=run_dir.name, val=history[-1]["val/spearman"], test=history[-1].get("test/spearman"),
                   best_val=max(h["val/spearman"] for h in history))
        rows.append(row)
        curves.append(pd.DataFrame(history).assign(run=run_dir.name))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(rows).sort_values("val", ascending=False), pd.concat(curves, ignore_index=True)


def plot_baseline(env, frame):
    """One figure per env: train loss, val loss, val Spearman, val RMSE / label std against epoch."""
    panels = [("train/loss", "train loss (normalised Δ)"), ("val/loss", "val loss (normalised Δ)"),
              ("val/spearman", "val Spearman"), ("rmse_over_std", "val RMSE / label std")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.9))
    for ax, (metric, title) in zip(axes, panels):
        for run, curve in frame.groupby("run"):
            cap, shuffled = curve.cap.iloc[0], curve.shuffled.iloc[0]
            ax.plot(curve.epoch, curve[metric], color=CAP_COLORS[cap], linestyle="--" if shuffled else "-",
                    linewidth=1.0 if shuffled else 1.6, label=f"Δ {cap}{' shuffled' if shuffled else ''}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8.5)
    caps = [cap for cap in CAP_COLORS if cap in set(frame.cap)]
    cap_names = {"d50": "max_delta = 50", "d100": "max_delta = 100", "full": "max_delta = null (full episode)"}
    colour_handles = [Line2D([], [], color=CAP_COLORS[cap], linewidth=2.5) for cap in caps]
    style_handles = [Line2D([], [], color=INK_2, linewidth=1.6), Line2D([], [], color=INK_2, linewidth=1.0,
                                                                        linestyle="--")]
    fig.legend(colour_handles, [cap_names[cap] for cap in caps], title="colour", loc="lower left",
               bbox_to_anchor=(0.12, 0), ncol=len(caps), fontsize=8, title_fontsize=8, labelcolor=INK_2)
    fig.legend(style_handles, ["solid: real labels", "dashed: shuffled labels (control)"], title="line",
               loc="lower right", bbox_to_anchor=(0.88, 0), ncol=2, fontsize=8, title_fontsize=8, labelcolor=INK_2)
    fig.suptitle(f"{env}: TRM head training curves (100k pairs, lr 1e-3, wd 1e-4, 20 epochs); val Δ range = the "
                 "run's cap, so values across caps are not directly comparable", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(OUT / f"baseline_curves_{env}.png", dpi=150)
    plt.close(fig)


def encoding_legend(target, table, colour_loc, marker_loc, ncol=3):
    """Two legends shared by the sweep figures: colour = lr, marker = n_pairs (on a figure or an axis)."""
    lr_handles = [Line2D([], [], color=LR_COLORS[lr], marker="o", linestyle="", markersize=7)
                  for lr in sorted(table.lr.unique())]
    pair_handles = [Line2D([], [], color=INK_2, marker=PAIR_MARKERS[k], linestyle="", markersize=7)
                    for k in sorted(table.n_pairs_k.unique())]
    common = dict(fontsize=8, title_fontsize=8, labelcolor=INK_2)
    colour = target.legend(lr_handles, [f"lr {lr:g}" for lr in sorted(table.lr.unique())], title="colour",
                           ncol=ncol, **colour_loc, **common)
    if hasattr(target, "add_artist"):                     # an axis keeps only its last legend unless re-added
        target.add_artist(colour)
    target.legend(pair_handles, [f"{k}k pairs" for k in sorted(table.n_pairs_k.unique())], title="marker",
                  ncol=min(ncol, 2), **marker_loc, **common)


def plot_val_vs_test(env, table):
    """Every config: final-epoch val vs test Spearman; top 5 by val numbered as in the curve figure."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for (lr, n_pairs_k), part in table.groupby(["lr", "n_pairs_k"]):
        ax.scatter(part.val, part.test, s=34, color=LR_COLORS[lr], marker=PAIR_MARKERS[n_pairs_k],
                   edgecolor="#fcfcfb", linewidth=0.8)
    # the top 5 sit in one tight cluster, so they get a ring instead of rank labels (ranks: top-curves figure)
    top = table.head(5)
    ax.scatter(top.val, top.test, s=110, facecolor="none", edgecolor=INK_2, linewidth=1.0)
    ax.annotate("ringed: top 5 by val\n(ranked in the top-curves figure)", (top.val.min(), top.test.min()),
                fontsize=8, color=INK_2, xytext=(-10, -30), textcoords="offset points", ha="right")
    low, high = table[["val", "test"]].min().min(), table[["val", "test"]].max().max()
    ax.plot([low, high], [low, high], color=MUTED, linewidth=0.8, linestyle=":")
    ax.set(xlabel="val Spearman (final epoch, used for selection)", ylabel="test Spearman (held-out episodes)")
    ax.set_title(f"{env}: {len(table)} configs, val vs test (dotted: y = x)", fontsize=10)
    encoding_legend(ax, table, dict(loc="upper left"), dict(loc="upper left", bbox_to_anchor=(0, 0.8)), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / f"sweep_{env}_val_vs_test.png", dpi=150)
    plt.close(fig)


def plot_top_curves(env, table, curves):
    """Val Spearman per epoch for the top 5 configs and the default; labels list only what differs from #1."""
    best = table.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    handles, labels = [], []
    for rank, (_, row) in enumerate(table.head(5).iterrows(), start=1):
        curve = curves[curves.run == row.run]
        line, = ax.plot(curve.epoch, curve["val/spearman"], color=CATEGORICAL[rank - 1], linewidth=2)
        handles.append(line)
        labels.append(f"#1  {describe(row)}  (test {row.test:.3f})" if rank == 1
                      else f"#{rank}  vs #1: {describe(row, best)}  (test {row.test:.3f})")
    if DEFAULT_RUN in set(table.run) and DEFAULT_RUN not in set(table.run.head(5)):
        row = table[table.run == DEFAULT_RUN].iloc[0]
        curve = curves[curves.run == DEFAULT_RUN]
        line, = ax.plot(curve.epoch, curve["val/spearman"], color=MUTED, linewidth=1.6, linestyle="--")
        handles.append(line)
        labels.append(f"default  vs #1: {describe(row, best)}  (test {row.test:.3f})")
    ax.set(xlabel="epoch", ylabel="val Spearman")
    ax.set_title(f"{env}: top 5 configs by final val Spearman, and the default config", fontsize=10)
    ax.legend(handles, labels, loc="lower right", fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / f"sweep_{env}_top_curves.png", dpi=150)
    plt.close(fig)


def plot_marginals(env, table):
    """One panel per hyperparameter: every config's val Spearman at each value, bar = mean over the rest."""
    fig, axes = plt.subplots(1, len(HPARAMS), figsize=(16, 4.4), sharey=True)
    for ax, name in zip(axes, HPARAMS):
        values = sorted(table[name].unique())
        for position, value in enumerate(values):
            part = table[table[name] == value]
            jitter = (part.n_pairs_k == 1000) * 0.12 - 0.06            # 100k left, 1M right of the tick
            for (lr, n_pairs_k), group in part.groupby(["lr", "n_pairs_k"]):
                ax.scatter(position + jitter[group.index], group.val, s=24, color=LR_COLORS[lr],
                           marker=PAIR_MARKERS[n_pairs_k], edgecolor="#fcfcfb", linewidth=0.6)
            ax.hlines(part.val.mean(), position - 0.3, position + 0.3, color=INK_2, linewidth=2)
        ax.set_xticks(range(len(values)), [HPARAM_TEXT[name](value) for value in values], fontsize=8)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("val Spearman (final epoch)")
    fig.suptitle(f"{env}: effect of each hyperparameter (dots = configs, bar = mean over the other settings)",
                 fontsize=10.5)
    encoding_legend(fig, table, dict(loc="lower left", bbox_to_anchor=(0.1, 0)),
                    dict(loc="lower right", bbox_to_anchor=(0.9, 0)))
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(OUT / f"sweep_{env}_marginals.png", dpi=150)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    for env in ENVS:
        frame = load_baseline(env)
        if not frame.empty:
            plot_baseline(env, frame)
        table, curves = load_sweep(env)
        if table.empty:
            continue
        table.to_csv(OUT / f"sweep_{env}.csv", index=False)
        print(f"== {env}: {len(table)} configs; top 5 by val")
        print(table.head(5)[["run", "val", "test", "best_val"]].to_string(index=False))
        plot_val_vs_test(env, table)
        plot_top_curves(env, table, curves)
        plot_marginals(env, table)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
