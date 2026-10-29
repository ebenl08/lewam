"""Latent diagnosis of the traced AnyGoal runs: E2-E4 (encoder), P1-P4 (predictor), S1 (search), variant - original.

  python analysis/diagnose/latent_diagnosis.py

Reads the traced runs (scripts/diagnose/eval_lewam_trace.py) and the latent banks (scripts/diagnose/build_latent_bank.py);
writes tables and plots to data/beam_smoke/analysis/latent_diagnosis/. Plan: notes/archive/latent-diagnosis-plan.md; results: notes/lewam/latent-diagnosis_1029.md.
"""

import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "analysis" / "planner_comparison"))
from plot_style import INK_2, MUTED, style  # noqa: E402

from lewam.diagnose.adapters import ADAPTERS  # noqa: E402
from lewam.diagnose.references import LatentBank  # noqa: E402
from lewam.diagnose.trace import predicted_step  # noqa: E402

SMOKE = LW / "data" / "beam_smoke"
BANKS = SMOKE / "latent_banks"
OUT = SMOKE / "analysis" / "latent_diagnosis"
DATASETS = Path.home() / ".stable_worldmodel" / "datasets"
SEED, N_BOOT, N_REF, K = 8902, 10_000, 2000, 5
MIN_GROUP = 5   # failure- or success-conditioned metrics need at least this many episodes in the group

TRAIN_BANKS = {"lewam-pusht": "lewam-pusht__pusht_expert_train", "lewam-cube": "lewam-cube__cube"}
RUNS = {  # key: env folder, traced group, dataset, checkpoint, task bank (None = training half 1), label
    "pusht_ga": ("pusht", "goal_anchored_trace", "pusht_expert_train", "lewam-pusht", None, "PushT (GA)"),
    "stackt_ga": ("stackt", "goal_anchored_trace", "stackt_test", "lewam-pusht", "lewam-pusht__stackt_test",
                  "StackT (GA)"),
    "cube_sa": ("cube", "start_anchored_trace", "cube", "lewam-cube", None, "Cube (SA)"),
    "cube_diagonal_sa": ("cube_diagonal", "start_anchored_c3_trace", "cube_diagonal_test", "lewam-cube",
                         "lewam-cube__cube_diagonal_test", "CubeDiagonal (SA)"),
    "cube_ra": ("cube", "anygoal_trace", "cube", "lewam-cube", None, "Cube (RA)"),
    "cube_hop_ra": ("cube_hop", "anygoal_c3_trace", "cube_hop_test", "lewam-cube", "lewam-cube__cube_hop_test",
                    "CubeHop (RA)"),
}
PAIRS = [  # name, original run, variant run, draw text
    ("StackT − PushT", "pusht_ga", "stackt_ga", "goal-anchored (GA), offset 50 / budget 100"),
    ("CubeDiagonal − Cube", "cube_sa", "cube_diagonal_sa", "start-anchored (SA), offset 50 / budget 100"),
    ("CubeHop − Cube", "cube_ra", "cube_hop_ra", "random start (RA), offset 25 / budget 50"),
]
GAP_METRICS = ["E2", "E3", "E4", "P1", "P2", "P3", "P4"]
COLORS = {  # PushT family blue, Cube family green; original dark, variant light
    "pusht_ga": "#1f5bb5", "stackt_ga": "#6ea8ee",
    "cube_sa": "#0b6b47", "cube_diagonal_sa": "#1baf7a", "cube_ra": "#0b6b47", "cube_hop_ra": "#1baf7a",
}
FORMULAS = {  # plot labels: each metric as its formula (matplotlib mathtext)
    "E2": r"$\rho_s\left(c(z_t),\ \mathrm{err}(t)\right)$",
    "E2_term": r"$\rho_s\left(c(z_t),\ e_j(t) / \tau_j\right)$",
    "E3": r"$\mathrm{pull}(z_t) - \overline{\mathrm{pull}}_{\mathrm{expert}}$",
    "E4": r"$\mathrm{ood}_{\mathrm{train}}(z_t)$",
    "P1": r"$\frac{1}{D}\,\Vert \hat{z}_{r,k} - z_{t_r + 5(k+1)} \Vert^2$",
    "P2": r"$c(\hat{z}_{r,k^\star}) - c(z_{t^\star})$",
    "P3": r"$\mathrm{pull}(\hat{z}_{r,k}) - \mathrm{pull}(z_{t_r + 5(k+1)})$",
    "P4": r"$\mathrm{ood}_{\mathrm{train}}(\hat{z}_{r,k}) - \mathrm{ood}_{\mathrm{train}}(z_{t_r + 5(k+1)})$",
    "pull": r"$\mathrm{pull}(z) = d_{\mathrm{task}}(z) - d_{\mathrm{train}}(z)$",
    "ood": r"$\mathrm{ood}_{\mathrm{train}}(z)$",
    "cost": r"$c(z) = \frac{1}{D}\,\Vert z - z_G \Vert^2$",
}
NAMES = {  # plot labels: the metric's name, shown above its formula
    "E2": "Goal cost vs task error (Spearman)",
    "E2_term": "Goal cost vs each task-error term (Spearman)",
    "E3": "Real-state pull, relative to expert",
    "E4": "Real-state distance to training data",
    "P1": "Latent prediction error",
    "P2": "Hallucination gap at the deadline",
    "P3": "Imagined pull beyond real",
    "P4": "Imagined − real training distance",
    "pull": "Pull toward training data",
    "ood": "Distance to training data",
    "cost": "Goal cost",
}


def label(key: str) -> str:
    """Metric name above its formula."""
    return f"{NAMES[key]}\n{FORMULAS[key]}"
CAPTION = "seed 8902 × 30 episodes, native_grad; shaded bands ±2 SD over episodes (one seed, so no seed-level spread)"
E2_ROWS = {}    # per real-row table: env -> (goal cost, task error) arrays, for the E2 bootstrap


def read_results(run_dir: Path) -> tuple[np.ndarray, list[int], list[int]]:
    text = (run_dir / "results.txt").read_text()
    successes = re.search(r"successes': array\(\[(.*?)\]\)", text, re.S).group(1).split(",")
    episodes = json.loads(re.search(r"episodes: (\[.*?\])", text).group(1))
    starts = json.loads(re.search(r"starts: (\[.*?\])", text).group(1))
    return np.array([s.strip() == "True" for s in successes]), episodes, starts


def bank_distance(bank: LatentBank, z: torch.Tensor, episodes: np.ndarray, dataset: str) -> np.ndarray:
    """kNN distance of each row of z to bank; a row's own episode is excluded when bank holds the same dataset."""
    if bank.dataset != dataset:
        return bank.knn(z, k=K).cpu().numpy()
    out = np.empty(len(z))
    for episode in np.unique(episodes):
        rows = episodes == episode
        out[rows] = bank.knn(z[torch.as_tensor(rows)], exclude_episode=int(episode), k=K).cpu().numpy()
    return out


def reference_sample(bank: LatentBank) -> tuple[torch.Tensor, np.ndarray]:
    """The fixed N_REF-frame sample of a bank used for the M2 check, with its episodes."""
    rows = torch.randperm(len(bank.latents), generator=torch.Generator().manual_seed(0))[:N_REF]
    return bank.latents[rows], bank.episodes[rows].numpy()


class Scorer:
    """pull and ood_train of latents for one run, with the run's two banks and expert baseline."""

    def __init__(self, banks: dict, dataset: str, checkpoint: str, task_bank: str | None, device: str):
        train_name = TRAIN_BANKS[checkpoint]
        self.train = banks[f"{train_name}__half0"]
        self.task = banks[task_bank or f"{train_name}__half1"]
        self.dataset, self.device = dataset, device

        held_out, held_out_episodes = reference_sample(banks[f"{train_name}__half1"])
        self.ood_scale = float(np.median(bank_distance(self.train, held_out.to(device), held_out_episodes,
                                                       self.task.dataset)))
        expert, expert_episodes = reference_sample(self.task)
        expert = expert.to(device)
        expert_pull = (bank_distance(self.task, expert, expert_episodes, self.task.dataset)
                       - bank_distance(self.train, expert, expert_episodes, self.task.dataset))
        self.expert_pull = float(np.median(expert_pull))

    def score(self, z: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pull, ood_train) of latents z (N, D) from the given dataset episodes."""
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        d_task = bank_distance(self.task, z, episodes, self.dataset)
        d_train = bank_distance(self.train, z, episodes, self.dataset)
        return d_task - d_train, d_train / self.ood_scale


def load_run(key: str, banks: dict, device: str) -> dict:
    env, group, dataset, checkpoint, task_bank, label = RUNS[key]
    run_dir = SMOKE / env / group / "native_grad" / str(SEED)
    config = json.loads((run_dir / "config.json").read_text())
    adapter = ADAPTERS[config["world"]["env_name"]]()
    offset = int(config["eval"]["goal_offset_steps"])
    successes, episodes, starts = read_results(run_dir)
    trace, snapshots = np.load(run_dir / "trace.npz"), np.load(run_dir / "snapshots.npz")
    goal_z = dict(zip(trace["goal_env"].tolist(), trace["goal_z"]))
    frameskip, rollout_steps = int(trace["frameskip"]), int(trace["rollout_steps"])
    scorer = Scorer(banks, dataset, checkpoint, task_bank, device)

    with h5py.File(DATASETS / f"{dataset}.h5", "r") as data:
        ep_offset = data["ep_offset"][:]
        start_rows = [int(ep_offset[ep]) + start for ep, start in zip(episodes, starts)]
        start_state = [adapter.goal(data, row) for row in start_rows]
        goal_state = [adapter.goal(data, row + offset) for row in start_rows]

    real = pd.DataFrame({"env": trace["real_env"], "step": trace["real_step"]})
    real["episode"] = [episodes[env] for env in real.env]
    real["success"] = [bool(successes[env]) for env in real.env]
    real_z = trace["real_z"]
    real["c"] = [((z - goal_z[env]) ** 2).mean() for z, env in zip(real_z, real.env)]
    states = [start_state[env] if step == 0 else snapshots[f"env_{env}"][step - 1]
              for env, step in zip(real.env, real.step)]
    term_errors = np.array([adapter.errors(state, goal_state[env]) for state, env in zip(states, real.env)])
    real["err"] = term_errors.max(1)
    for term, column in zip(adapter.terms, term_errors.T):
        real[f"term:{term}"] = column
    real["pull"], real["ood"] = scorer.score(real_z, real.episode.to_numpy())

    imag = pd.DataFrame({"env": trace["imag_env"], "replan_step": trace["imag_step"], "k": trace["imag_k"],
                         "steps_to_goal": trace["imag_steps_to_goal"]})
    imag["episode"] = [episodes[env] for env in imag.env]
    imag["success"] = [bool(successes[env]) for env in imag.env]
    imag["step"] = [predicted_step(t, k, frameskip) for t, k in zip(imag.replan_step, imag.k)]
    deadline_k = np.clip(np.round(imag.steps_to_goal), 1, rollout_steps) - 1
    imag["deadline"] = imag.k == deadline_k
    imag_z = trace["imag_z"]
    imag["c_hat"] = [((z - goal_z[env]) ** 2).mean() for z, env in zip(imag_z, imag.env)]
    imag["pull_hat"], imag["ood_hat"] = scorer.score(imag_z, imag.episode.to_numpy())

    real_index = {(env, step): row for row, (env, step) in enumerate(zip(real.env, real.step))}
    matched = [real_index.get((env, step), -1) for env, step in zip(imag.env, imag.step)]
    imag["real_row"] = matched
    has_real = imag.real_row >= 0
    imag["latent_err"] = np.nan
    imag.loc[has_real, "latent_err"] = ((imag_z[has_real.to_numpy()]
                                         - real_z[imag.real_row[has_real].to_numpy()]) ** 2).mean(-1)
    for column in ("c", "pull", "ood"):
        imag[f"{column}_real"] = np.nan
        imag.loc[has_real, f"{column}_real"] = real[column].to_numpy()[imag.real_row[has_real].to_numpy()]
    real["run"], imag["run"] = key, key
    return dict(key=key, label=label, successes=successes, real=real, imag=imag, scorer=scorer, terms=adapter.terms,
                expert_pull=scorer.expert_pull, ood_scale=scorer.ood_scale)


def episode_table(run: dict) -> pd.DataFrame:
    """Per-episode summaries of the mean-type metrics."""
    real, imag = run["real"], run["imag"]
    matched = imag.dropna(subset=["latent_err"])
    at_deadline = matched[matched.deadline]
    table = pd.DataFrame(index=pd.Index(sorted(real.env.unique()), name="env"))
    table["success"] = run["successes"][table.index]
    table["E3"] = (real.pull - run["expert_pull"]).groupby(real.env).mean()
    table["E4"] = real.ood.groupby(real.env).mean()
    table["P1"] = matched.latent_err.groupby(matched.env).mean()
    table["P2"] = (at_deadline.c_hat - at_deadline.c_real).groupby(at_deadline.env).mean()
    table["P3"] = (matched.pull_hat - matched.pull_real).groupby(matched.env).mean()
    table["P4"] = (matched.ood_hat - matched.ood_real).groupby(matched.env).mean()
    table["min_c_real"] = real.c.groupby(real.env).min()
    deadline_rows = imag[imag.deadline]
    table["min_c_hat"] = deadline_rows.c_hat.groupby(deadline_rows.env).min()
    return table


def e2(real: pd.DataFrame, envs: np.ndarray | None = None) -> float:
    """Spearman rho of real goal cost vs task error over the rows of the given envs (all if None)."""
    if envs is None:
        return float(spearmanr(real.c, real.err).statistic)
    if id(real) not in E2_ROWS:
        E2_ROWS[id(real)] = {env: (rows.c.to_numpy(), rows.err.to_numpy()) for env, rows in real.groupby("env")}
    by_env = E2_ROWS[id(real)]
    cost = np.concatenate([by_env[env][0] for env in envs])
    err = np.concatenate([by_env[env][1] for env in envs])
    return float(spearmanr(cost, err).statistic)


def term_correlations(run: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Spearman rho of real goal cost vs each task-error term; 95% CI from a bootstrap over episodes."""
    real = run["real"]
    envs = np.sort(real.env.unique())
    picks = rng.integers(0, len(envs), (N_BOOT, len(envs)))
    rows = []
    for term in run["terms"]:
        column = f"term:{term}"
        by_env = {env: (group.c.to_numpy(), group[column].to_numpy()) for env, group in real.groupby("env")}
        point = float(spearmanr(real.c, real[column]).statistic)
        samples = []
        for pick in picks:
            chosen = envs[pick]
            cost = np.concatenate([by_env[env][0] for env in chosen])
            term_error = np.concatenate([by_env[env][1] for env in chosen])
            samples.append(spearmanr(cost, term_error).statistic)
        low, high = np.nanpercentile(samples, [2.5, 97.5])
        rows.append(dict(run=run["key"], label=run["label"], term=term, rho=point, ci_low=low, ci_high=high,
                         n_frames=len(real), n_episodes=len(envs)))
    return pd.DataFrame(rows)


def run_metrics(run: dict, table: pd.DataFrame) -> dict:
    successes = run["successes"]
    failed, succeeded = table[~table.success], table[table.success]
    metrics = dict(run=run["key"], label=run["label"], success_rate=100 * successes.mean(),
                   n_failures=int((~successes).sum()), E2=e2(run["real"]),
                   expert_pull=run["expert_pull"], ood_scale=run["ood_scale"])
    for name in ("E3", "E4", "P1", "P2", "P3", "P4"):
        metrics[name] = float(table[name].mean())
    enough = len(failed) >= MIN_GROUP and len(succeeded) >= MIN_GROUP
    metrics["P2_failures"] = float(failed.P2.mean()) if len(failed) >= MIN_GROUP else np.nan
    metrics["S1"] = float(failed.min_c_hat.median() - succeeded.min_c_real.median()) if enough else np.nan
    metrics["success_floor"] = float(succeeded.min_c_real.median()) if len(succeeded) >= MIN_GROUP else np.nan
    return metrics


def pair_gaps(runs: dict, tables: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Variant - original per metric, 95% CI from a bootstrap over episodes (each run resampled on its own)."""
    rows = []
    for name, original, variant, draw in PAIRS:
        n_orig, n_var = len(tables[original]), len(tables[variant])
        pick_orig = rng.integers(0, n_orig, (N_BOOT, n_orig))
        pick_var = rng.integers(0, n_var, (N_BOOT, n_var))
        for metric in GAP_METRICS:
            if metric == "E2":
                point = e2(runs[variant]["real"]) - e2(runs[original]["real"])
                env_orig, env_var = tables[original].index.to_numpy(), tables[variant].index.to_numpy()
                samples = np.array([e2(runs[variant]["real"], env_var[pick_var[b]])
                                    - e2(runs[original]["real"], env_orig[pick_orig[b]]) for b in range(N_BOOT)])
            else:
                values_orig, values_var = tables[original][metric].to_numpy(), tables[variant][metric].to_numpy()
                counts = (np.isfinite(values_orig).sum(), np.isfinite(values_var).sum())
                if min(counts) < MIN_GROUP:  # e.g. P2 on an original whose episodes end before the deadline
                    rows.append(dict(pair=name, draw=draw, metric=metric, gap=np.nan, ci_low=np.nan, ci_high=np.nan,
                                     n_original=counts[0], n_variant=counts[1]))
                    continue
                point = np.nanmean(values_var) - np.nanmean(values_orig)
                samples = np.nanmean(values_var[pick_var], 1) - np.nanmean(values_orig[pick_orig], 1)
            low, high = np.nanpercentile(samples, [2.5, 97.5])
            n_orig, n_var = (len(tables[original]), len(tables[variant])) if metric == "E2" else counts
            rows.append(dict(pair=name, draw=draw, metric=metric, gap=point, ci_low=low, ci_high=high,
                             n_original=n_orig, n_variant=n_var))
    return pd.DataFrame(rows)


def mean_band(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Mean and ±2 SD over episodes of per-episode means of y at each x."""
    per_episode = frame.groupby([x, "env"])[y].mean().reset_index()
    return per_episode.groupby(x)[y].agg(["mean", "std", "count"]).fillna(0.0)


def plot_band(ax, band: pd.DataFrame, color: str, label: str, dashed: bool = False) -> None:
    ax.plot(band.index, band["mean"], color=color, lw=1.6, ls="--" if dashed else "-", label=label)
    ax.fill_between(band.index, band["mean"] - 2 * band["std"], band["mean"] + 2 * band["std"], color=color,
                    alpha=0.12, lw=0)


def plot_p1(runs: dict) -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.2 * len(PAIRS), 3.4))
    for ax, (name, original, variant, draw) in zip(axes, PAIRS):
        for key in (original, variant):
            matched = runs[key]["imag"].dropna(subset=["latent_err"])
            plot_band(ax, mean_band(matched, "k", "latent_err"), COLORS[key], runs[key]["label"])
        ax.set(title=f"{name}\n{draw}", xlabel=r"imagined depth $k$ (0-based)", ylabel=label("P1"))
        ax.legend(fontsize=8)
    fig.suptitle(f"Imagined vs real latent error by depth — {CAPTION}", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "p1_latent_error_by_depth.png", dpi=150)
    plt.close(fig)


def plot_cost(runs: dict) -> None:
    keys = [key for _, original, variant, _ in PAIRS for key in (original, variant)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.T.flat, keys):
        run = runs[key]
        for success, shade in ((True, COLORS[key]), (False, MUTED)):
            group = "success" if success else "failure"
            n = int((run["successes"] == success).sum())
            if n == 0:
                continue
            real = run["real"][run["real"].success == success]
            imag = run["imag"][(run["imag"].success == success) & run["imag"].deadline]
            plot_band(ax, mean_band(real, "step", "c"), shade, rf"real $c(z_t)$, {group} (n={n})")
            plot_band(ax, mean_band(imag, "replan_step", "c_hat"), shade,
                      rf"imagined $c(\hat{{z}}_{{r,k^\star}})$, {group}", dashed=True)
        ax.set(title=run["label"], xlabel=r"env step $t$", ylabel=label("cost"))
        ax.legend(fontsize=7)
    fig.suptitle(f"Goal cost: real frames vs imagined deadline latent (x = replan step) — {CAPTION}",
                 fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "cost_real_vs_imagined.png", dpi=150)
    plt.close(fig)


def plot_pull_ood(runs: dict) -> None:
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(4.4 * len(PAIRS), 6.6))
    for col, (name, original, variant, draw) in enumerate(PAIRS):
        for row, (real_col, imag_col, title) in enumerate((("pull", "pull_hat", label("pull")),
                                                           ("ood", "ood_hat", label("ood")))):
            ax = axes[row, col]
            for key in (original, variant):
                run = runs[key]
                plot_band(ax, mean_band(run["real"], "step", real_col), COLORS[key], f"{run['label']} real")
                plot_band(ax, mean_band(run["imag"], "step", imag_col), COLORS[key], f"{run['label']} imagined",
                          dashed=True)
                baseline = run["expert_pull"] if real_col == "pull" else 1.0
                ax.axhline(baseline, color=COLORS[key], lw=0.8, ls=":")
            ax.set(title=f"{name}\n{draw}", xlabel="env step (imagined: predicted step)", ylabel=title)
            ax.legend(fontsize=7)
    fig.suptitle(r"Real $z_t$ vs imagined $\hat{z}_{r,k}$; dotted = $\overline{\mathrm{pull}}_{\mathrm{expert}}$ "
                 rf"(top) and in-distribution level 1 (bottom) — {CAPTION}", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "pull_ood_real_vs_imagined.png", dpi=150)
    plt.close(fig)


def plot_term_correlations(terms: pd.DataFrame, runs: dict) -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.6 * len(PAIRS), 3.8))
    for ax, (name, original, variant, draw) in zip(axes, PAIRS):
        term_names = list(runs[original]["terms"])
        y = np.arange(len(term_names))
        for offset, key in ((0.12, original), (-0.12, variant)):
            rows = terms[terms.run == key].set_index("term").loc[term_names]
            ax.errorbar(rows.rho, y + offset, xerr=[rows.rho - rows.ci_low, rows.ci_high - rows.rho], fmt="o",
                        color=COLORS[key], ecolor=COLORS[key], capsize=3, label=runs[key]["label"])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set(title=f"{name}\n{draw}", yticks=y, yticklabels=term_names, xlim=(-1.05, 1.05),
               xlabel=FORMULAS["E2_term"])
        ax.legend(fontsize=8)
    fig.suptitle(f"{NAMES['E2_term']}; error bars 95% bootstrap CI over episodes (10,000 resamples), "
                 "seed 8902 × 30, real frames at each replan", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "e2_per_term.png", dpi=150)
    plt.close(fig)


def plot_gaps(gaps: pd.DataFrame) -> None:
    """Encoder metrics on the top row, predictor metrics on the bottom row; metrics with no gap are left out."""
    plotted = [metric for metric in GAP_METRICS if gaps[gaps.metric == metric].gap.notna().any()]
    rows_of_metrics = [[metric for metric in plotted if metric.startswith(prefix)] for prefix in ("E", "P")]
    n_cols = max(len(row) for row in rows_of_metrics)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 7.2), sharey=True)
    for ax in axes[0, len(rows_of_metrics[0]):]:
        ax.set_visible(False)
    pair_names = [name for name, *_ in PAIRS]
    placed = [(axes[row, col], metric) for row, metrics in enumerate(rows_of_metrics)
              for col, metric in enumerate(metrics)]
    for ax, metric in placed:
        rows = gaps[gaps.metric == metric].set_index("pair").loc[pair_names]
        y = np.arange(len(pair_names))
        ax.errorbar(rows.gap, y, xerr=[rows.gap - rows.ci_low, rows.ci_high - rows.gap], fmt="o", color="#1f5bb5",
                    ecolor=INK_2, capsize=3)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(label(metric), fontsize=10)
        ax.set(yticks=y, yticklabels=pair_names)
    fig.suptitle("Variant − original per metric; error bars 95% bootstrap CI over episodes (10,000 resamples), "
                 "seed 8902 × 30", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "pair_gaps.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    banks = {path.stem: LatentBank.load(path) for path in BANKS.glob("*.pt")}
    runs = {key: load_run(key, banks, device) for key in RUNS}
    tables = {key: episode_table(run) for key, run in runs.items()}

    metrics = pd.DataFrame([run_metrics(runs[key], tables[key]) for key in RUNS])
    gaps = pair_gaps(runs, tables, np.random.default_rng(0))
    term_rng = np.random.default_rng(1)
    terms = pd.concat([term_correlations(runs[key], term_rng) for key in RUNS])
    p1_depth = pd.concat([mean_band(run["imag"].dropna(subset=["latent_err"]), "k", "latent_err").assign(run=key)
                          for key, run in runs.items()])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    gaps.to_csv(OUT / "pair_gaps.csv", index=False)
    terms.to_csv(OUT / "e2_per_term.csv", index=False)
    p1_depth.to_csv(OUT / "p1_by_depth.csv")
    pd.concat([run["real"] for run in runs.values()]).to_csv(OUT / "rows_real.csv", index=False)
    pd.concat([run["imag"] for run in runs.values()]).to_csv(OUT / "rows_imag.csv", index=False)

    plot_p1(runs)
    plot_cost(runs)
    plot_pull_ood(runs)
    plot_gaps(gaps)
    plot_term_correlations(terms, runs)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(metrics.drop(columns=["label"]))
        print(gaps)
        print(terms.drop(columns=["label"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
