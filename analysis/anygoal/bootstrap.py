"""AnyGoal originals vs variants with LeWAM, goal-MSE objective: success rate per planner, hierarchical bootstrap.

Bootstrap: resample seeds with replacement, then episodes with replacement inside each drawn seed (10,000 resamples,
percentile 95% CI). Variant − original differences are unpaired (different episodes), so each env is resampled on
its own draw. Per-episode successes come from `episode_successes` in results.txt. Episode draws: RA = random start,
offset 25 / budget 50; GA = goal-anchored, SA = start-anchored, both offset 50 / budget 100 (hatched bars). Reads
data/beam_smoke/<env>/<group>/<arm>/<seed>/ (see RUNS), writes data/beam_smoke/analysis/anygoal/. Run from playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/anygoal/bootstrap.py
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "analysis" / "planner_comparison"))
from plot_style import INK_2, style  # noqa: E402

SMOKE = LW / "data" / "beam_smoke"
OUT = SMOKE / "analysis" / "anygoal"
SEEDS = [8902, 10748, 17401, 18714, 28967, 30531, 36136, 37182, 47259, 60813]
N_BOOT, RNG_SEED = 10_000, 0
FAMILIES = {   # RA original first, then its RA variants, then the anchored original and variant
    "pusht": ["pusht", "avoidt", "stackt", "pusht_antigoal", "pusht_ga", "stackt_ga"],
    "tworoom": ["tworoom", "tworoom_route"],
    "cube": ["cube", "cube_hop", "cube_diagonal", "cube_sa", "cube_diagonal_sa"],
}
ENV_TEXT = {"pusht": "PushT (original, RA)", "avoidt": "AvoidT (RA)", "stackt": "StackT (RA)",
            "pusht_antigoal": "PushT-Antigoal (RA)", "pusht_ga": "PushT (original, GA)", "stackt_ga": "StackT (GA)",
            "tworoom": "TwoRoom (original, RA)", "tworoom_route": "TwoRoom-Route (RA)",
            "cube": "Cube (original, RA)", "cube_hop": "CubeHop (RA)", "cube_diagonal": "CubeDiagonal (RA)",
            "cube_sa": "Cube (original, SA)", "cube_diagonal_sa": "CubeDiagonal (SA)"}
# bar -> (env folder, run group); other bars read <env>/anygoal/. The Cube variants use AnyGoal's criterion 3
# (scripts/anygoal/eval_lewam_anygoal_cube.py)
RUNS = {"cube_hop": ("cube_hop", "anygoal_c3"), "cube_diagonal": ("cube_diagonal", "anygoal_c3"),
        "pusht_ga": ("pusht", "goal_anchored"), "stackt_ga": ("stackt", "goal_anchored"),
        "cube_sa": ("cube", "start_anchored"), "cube_diagonal_sa": ("cube_diagonal", "start_anchored_c3")}
ANCHORED = {"pusht_ga", "stackt_ga", "cube_sa", "cube_diagonal_sa"}   # offset 50 / budget 100
ORIGINAL = {"stackt_ga": "pusht_ga", "cube_diagonal_sa": "cube_sa"}   # else the family's first bar
DRAW_TEXT = {"pusht": "RA: random start, offset 25 / budget 50. GA (hatched): goal-anchored, goal = demo's last frame, "
                      "offset 50 / budget 100",
             "tworoom": "RA: random start, offset 25 / budget 50",
             "cube": "RA: random start, offset 25 / budget 50. SA (hatched): start-anchored, start = frame 0, "
                     "offset 50 / budget 100"}
ARMS = ["native_grad", "chains_b32c1", "topk_goal", "soft_t1_goal"]
ARM_TEXT = {"native_grad": "native gradient", "chains_b32c1": "independent chains\n(B32 C1)",
            "topk_goal": "top-k beam\n(B8 C4)", "soft_t1_goal": "soft beam τ=1\n(B8 C4)"}
# categorical slots in fixed order (dataviz palette, CVD-checked): original = slot 1, RA variants = slots 2..4
ENV_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RA_TWIN = {"pusht_ga": "pusht", "stackt_ga": "stackt", "cube_sa": "cube", "cube_diagonal_sa": "cube_diagonal"}


def env_color(env, envs):
    """An anchored bar takes its RA env's colour; the hatch marks the draw."""
    return ENV_SLOTS[envs.index(RA_TWIN.get(env, env))]


def episode_successes(run_dir):
    text = (run_dir / "results.txt").read_text()
    listed = re.search(r"'episode_successes': array\(\[([^\]]*)\]", text, re.S).group(1)
    return np.array([token.strip() == "True" for token in listed.replace("\n", " ").split(",")])


def load(env, arm):
    """(seeds, episodes) success matrix over the seeds that have finished, or None."""
    folder, group = RUNS.get(env, (env, "anygoal"))
    runs = [SMOKE / folder / group / arm / str(seed) for seed in SEEDS]
    runs = [run for run in runs if (run / "results.txt").exists()]
    return np.stack([episode_successes(run) for run in runs]) if runs else None


def boot_rates(success, rng):
    """(N_BOOT,) success rate of each hierarchical resample."""
    n_seeds, n_episodes = success.shape
    seed_draw = rng.integers(0, n_seeds, size=(N_BOOT, n_seeds))
    episode_draw = rng.integers(0, n_episodes, size=(N_BOOT, n_seeds, n_episodes))
    return 100 * success[seed_draw[:, :, None], episode_draw].mean(axis=(1, 2))


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    rows, diffs = [], []
    for family, envs in FAMILIES.items():
        for arm in ARMS:
            rates = {}
            for env in envs:
                success = load(env, arm)
                if success is None:
                    continue
                rates[env] = boot_rates(success, rng)
                low, high = np.percentile(rates[env], [2.5, 97.5])
                rows.append(dict(family=family, env=env, arm=arm, sr=100 * success.mean(), ci_low=low, ci_high=high,
                                 seeds=len(success), episodes=success.size))
            for env in envs[1:]:
                original = ORIGINAL.get(env, envs[0])
                if env in ANCHORED and env not in ORIGINAL:   # an anygoal original: nothing to compare with
                    continue
                if original in rates and env in rates:
                    delta = rates[env] - rates[original]
                    low, high = np.percentile(delta, [2.5, 97.5])
                    point = [r["sr"] for r in rows if r["env"] == env and r["arm"] == arm][0] \
                        - [r["sr"] for r in rows if r["env"] == original and r["arm"] == arm][0]
                    diffs.append(dict(family=family, env=env, arm=arm, minus_original=point, ci_low=low, ci_high=high))
    table, diff_table = pd.DataFrame(rows), pd.DataFrame(diffs)
    table.to_csv(OUT / "success_ci.csv", index=False)
    diff_table.to_csv(OUT / "minus_original_ci.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(table.to_string(index=False))
        print(diff_table.to_string(index=False))

    for family, envs in FAMILIES.items():
        part = table[table.family == family]
        present = [env for env in envs if env in set(part.env)]
        if not present:
            continue
        width = 0.8 / len(envs)
        fig, ax = plt.subplots(figsize=(2.4 * len(ARMS) + 1, 4.6))
        for slot, env in enumerate(envs):
            offset = (slot - (len(envs) - 1) / 2) * width
            for position, arm in enumerate(ARMS):
                row = part[(part.env == env) & (part.arm == arm)]
                if row.empty:
                    continue
                draw_bar(ax, position + offset, row.iloc[0], width, env_color(env, envs), env in ANCHORED)
        arm_axis(ax)
        seeds = sorted(set(part.seeds))
        ax.set_title(f"{family}: original vs AnyGoal variants, LeWAM goal-MSE objective\n"
                     f"{'/'.join(map(str, seeds))} seeds × 30 episodes, 95% hierarchical bootstrap CI\n"
                     f"{DRAW_TEXT[family]}", fontsize=9)
        handles = [Patch(facecolor=env_color(env, envs), hatch="///" if env in ANCHORED else None,
                         edgecolor="#fcfcfb", linewidth=0) for env in envs]
        ax.legend(handles, [ENV_TEXT[e] for e in envs], loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=min(len(envs), 3), fontsize=8.5, frameon=False, labelcolor=INK_2)
        fig.tight_layout()
        fig.savefig(OUT / f"{family}_original_vs_variants.png", dpi=150)
        plt.close(fig)
    anchored_plot(table, diff_table)
    for old in ("avoidt_planners.png", "stackt_planners.png", "minus_native_ci.csv", "anchored_original_vs_variant.png"):
        (OUT / old).unlink(missing_ok=True)
    print(f"wrote {OUT}")


def draw_bar(ax, x, row, width, color, hatched):
    """One success-rate bar with its CI whisker and value label."""
    ax.bar(x, row.sr, width * 0.92, color=color, hatch="///" if hatched else None,
           edgecolor="#fcfcfb", linewidth=0, yerr=[[row.sr - row.ci_low], [row.ci_high - row.sr]], capsize=2.5,
           error_kw=dict(elinewidth=0.9, ecolor=INK_2))
    short = row.sr < 10                                  # too short for a label inside: put it above
    ax.text(x, row.ci_high + 1.5 if short else 2, f"{row.sr:.0f}", ha="center", va="bottom", fontsize=7.5,
            color=INK_2 if short else "#fcfcfb", fontweight="bold")


def arm_axis(ax):
    """Planner groups on x, success rate on y."""
    for boundary in np.arange(0.5, len(ARMS) - 1):
        ax.axvline(boundary, color="#d6d5cd", linewidth=0.8)
    ax.set_xticks(range(len(ARMS)), [ARM_TEXT[a] for a in ARMS], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set(xlim=(-0.5, len(ARMS) - 0.5), ylim=(0, 115), ylabel="success rate (%)")


def anchored_plot(table, diff_table):
    """Anchored bars only, one figure per family: original vs variant with the variant − original gap."""
    figures = {"pusht_stackt_ga": ("pusht", "pusht_ga", "stackt_ga", "GA: goal-anchored, goal = demo's last frame"),
               "cube_cubediag_sa": ("cube", "cube_sa", "cube_diagonal_sa", "SA: start-anchored, start = frame 0 "
                                                                           "(Cube: LeWAM rule, CubeDiagonal: criterion 3)")}
    width = 0.36
    for name, (family, original, variant, draw) in figures.items():
        envs = FAMILIES[family]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for slot, env in enumerate((original, variant)):
            for position, arm in enumerate(ARMS):
                row = table[(table.env == env) & (table.arm == arm)]
                if not row.empty:
                    draw_bar(ax, position + (slot - 0.5) * width, row.iloc[0], width, env_color(env, envs), False)
        for position, arm in enumerate(ARMS):
            gap = diff_table[(diff_table.env == variant) & (diff_table.arm == arm)]
            if not gap.empty:
                gap = gap.iloc[0]
                ax.text(position, 104, f"Δ {gap.minus_original:+.1f}\n[{gap.ci_low:+.1f}, {gap.ci_high:+.1f}]",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)
        arm_axis(ax)
        seeds = sorted(set(table[table.env.isin([original, variant])].seeds))
        ax.set_title(f"{ENV_TEXT[original]} vs {ENV_TEXT[variant]}, LeWAM goal-MSE objective\n"
                     f"{draw}, offset 50 / budget 100\n{'/'.join(map(str, seeds))} seeds × 30 episodes, "
                     f"95% hierarchical bootstrap CI, Δ = variant − original", fontsize=9)
        handles = [Patch(facecolor=env_color(env, envs)) for env in (original, variant)]
        ax.legend(handles, [ENV_TEXT[original], ENV_TEXT[variant]], loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=2, fontsize=8.5, frameon=False, labelcolor=INK_2)
        fig.tight_layout()
        fig.savefig(OUT / f"anchored_{name}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
