"""Beam-tree latency on PushT beside the P2 arms: end-to-end batch-1 GPU ms per plan, amortized GPU ms per env plan.

Reads p2/latency/pusht/<arm>/h<h>/ and p2/sweeps/pusht/batch/<arm>/h5_b<B>/; writes beam_latency_e2e.png,
beam_latency_amortized.png and beam_latency.md to analysis/p2/. Mean over seeds, error bars ±2 SD over seeds.

CLI (from playground/lewam/):
    /home/eben/.conda/envs/lewam/bin/python analysis/planner_comparison/beam_latency.py
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from latency_compare import DATA, LATENCY, OUT, TITLES, ms_per_plan
from latency_sweeps import SEEDS, SWEEPS, call_stats
from plot_style import INK_2, SURFACE, color, style

BEAM_TITLES = {'beam_topk_goal': 'Beam top-k B8×C4', 'beam_soft_t3_goal': 'Beam soft τ=3 B8×C4',
               'beam_chains_b32c1': 'Beam chains B32×C1'}
TITLES = {**TITLES, **BEAM_TITLES}
ARMS = ['native', 'grid_w1', 'cem', 'leflow_cem', 'icem_faithful', 'leflow_icem', *BEAM_TITLES]
MARKERS = {'native': '^', 'grid_w1': 'o', 'cem': 's', 'leflow_cem': 'D', 'icem_faithful': 's', 'leflow_icem': 'D',
           'beam_topk_goal': 'v', 'beam_soft_t3_goal': 'P', 'beam_chains_b32c1': 'X'}
HORIZONS = (2, 5, 10)
BATCHES = (1, 2, 4, 8, 16, 32, 64)


def mean_2sd(values):
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return dict(mean=float(values.mean()), sd2=float(2 * sd), n=len(values))


def e2e_stats():
    """{(arm, h): mean ± 2 SD of batch-1 GPU ms per plan over seeds}."""
    stats = {}
    for arm in ARMS:
        for h in HORIZONS:
            paths = glob.glob(f'{LATENCY}/pusht/{arm}/h{h}/*/timing.json')
            if paths:
                stats[(arm, h)] = mean_2sd([ms_per_plan(json.loads(Path(p).read_text()))[0] for p in paths])
    return stats


def batch_runs(arm, batch):
    if batch == 1:
        pattern = f'{LATENCY}/pusht/{arm}/h5/*/timing.json'
    else:  # B = 32 reruns replace the first runs (latency_sweeps.batch_points)
        sweep = 'batch_rerun' if (SWEEPS / 'batch_rerun' / arm / f'h5_b{batch}').exists() else 'batch'
        pattern = f'{SWEEPS}/{sweep}/{arm}/h5_b{batch}/*/timing.json'
    return [p for p in glob.glob(pattern) if int(Path(p).parent.name) in SEEDS]


def amortized_stats():
    """{arm: {B: {ms_per_call, ms_per_env}: mean ± 2 SD over the 3 sweep seeds}}."""
    stats = {}
    for arm in ARMS:
        for batch in BATCHES:
            runs = [call_stats(json.loads(Path(p).read_text()), batch) for p in batch_runs(arm, batch)]
            if runs:
                stats.setdefault(arm, {})[batch] = {key: mean_2sd([r[key] for r in runs])
                                                    for key in ('ms_per_call', 'ms_per_env')}
    return stats


REFINE_STAGES = ('refine_setup', 'refine_forward', 'best_update', 'loss', 'zero_grad', 'backward', 'grad_clip',
                 'adam_step', 'action_clamp')  # native's Adam refinement, timed by the `planner` detail level
NATIVE_DETAIL = DATA / 'p2' / 'breakdown_detail' / 'pusht' / 'planner' / 'native'


def native_no_refine():
    """{h: batch-1 GPU ms per plan of LeWAM native minus its refinement stages, mean ± 2 SD over seeds}."""
    stats = {}
    for h in HORIZONS:
        values = []
        for path in glob.glob(f'{NATIVE_DETAIL}/h{h}/*/timing.json'):
            timing = json.loads(Path(path).read_text())
            refine = sum(timing['tree'][stage]['gpu_ms_per_plan'] for stage in REFINE_STAGES)
            values.append(ms_per_plan(timing)[0] - refine)
        if values:
            stats[h] = mean_2sd(values)
    return stats


def plot_e2e(stats):
    """Beam arms and LeWAM native without its refinement, all batch 1; P2 arms: pe2_latency_bars.png."""
    arms = list(BEAM_TITLES)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    width = 0.8 / (len(arms) + 1)  # last slot of each group: native minus refinement
    native = native_no_refine()
    for j, h in enumerate(HORIZONS):
        x = j + len(arms) * width - 0.4 + width / 2
        if h not in native:
            ax.text(x, 3, 'not timed', ha='center', va='bottom', rotation=90, fontsize=8, color=color('native'))
            continue
        s = native[h]
        ax.bar(x, s['mean'], width, yerr=s['sd2'], capsize=2, color=color('native'), edgecolor=SURFACE, linewidth=1,
               error_kw={'ecolor': INK_2}, label=None if j > HORIZONS.index(5) else
               f"LeWAM native minus Adam refinement ({s['n']} seeds, P2 stage timers, GPUs 3/4/6)")
        ax.text(x, (s['mean'] + s['sd2']) * 1.04, f"{s['mean']:.0f}", ha='center', va='bottom', fontsize=7,
                color=INK_2)
    for k, arm in enumerate(arms):
        cells = [(j, stats[(arm, h)]) for j, h in enumerate(HORIZONS) if (arm, h) in stats]
        xs = [j + k * width - 0.4 + width / 2 for j, _ in cells]
        ys = [s['mean'] for _, s in cells]
        ax.bar(xs, ys, width, yerr=[s['sd2'] for _, s in cells], capsize=2, label=TITLES[arm], color=color(arm),
               edgecolor=SURFACE, linewidth=1, error_kw={'ecolor': INK_2})
        for x, y, (_, s) in zip(xs, ys, cells):
            ax.text(x, (y + s['sd2']) * 1.04, f'{y:.0f}', ha='center', va='bottom', fontsize=7, color=INK_2)
        for j, h in enumerate(HORIZONS):
            if (arm, h) not in stats:
                x = j + k * width - 0.4 + width / 2
                ax.text(x, 3, 'not run', ha='center', va='bottom', rotation=90, fontsize=8, color=color(arm))
    ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.55, len(HORIZONS) - 0.45)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_xticks(range(len(HORIZONS)), [f'h = {h}' for h in HORIZONS])
    ax.set_ylabel('GPU ms per plan (encode + plan)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('PushT end-to-end latency, one env per plan call (batch 1). Beam: GPU 8, 10 seeds\n'
                 '(mean over seeds, error bars ±2 SD)', fontsize=10)
    fig.savefig(OUT / 'beam_latency_e2e.png', dpi=150)
    plt.close(fig)


def plot_amortized(stats):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for arm, points in stats.items():
        batches = list(points)
        for ax, key in zip(axes, ('ms_per_call', 'ms_per_env')):
            ax.errorbar(batches, [points[b][key]['mean'] for b in batches],
                        yerr=[points[b][key]['sd2'] for b in batches], color=color(arm), marker=MARKERS[arm],
                        capsize=2, label=TITLES[arm])
    for ax, label in zip(axes, ('GPU ms per plan call', 'GPU ms per env plan (amortized)')):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
        ax.set_xlabel('envs per plan call (B), h = 5')
        ax.set_ylabel(label)
    axes[1].legend(fontsize=7)
    fig.suptitle('PushT latency over batch size (mean over 3 seeds, error bars ±2 SD; '
                 'envs that finish drop out of later calls)', fontsize=10)
    fig.savefig(OUT / 'beam_latency_amortized.png', dpi=150)
    plt.close(fig)


def table(e2e, amortized):
    lines = ['# Beam-tree latency (PushT, GPU ms, mean ± 2 SD over seeds)', '',
             '## End-to-end, batch 1', '', '| Arm | h | ms per plan | ±2 SD | seeds | ratio to CEM same h |',
             '|---|---|---|---|---|---|']
    for (arm, h), s in e2e.items():
        cem = e2e.get(('cem', h))
        ratio = f"{s['mean'] / cem['mean']:.2f}" if cem else '—'
        lines.append(f"| {TITLES[arm]} | {h} | {s['mean']:.1f} | {s['sd2']:.1f} | {s['n']} | {ratio} |")
    lines += ['', '## Amortized over batch, h = 5', '',
              '| Arm | B | ms per call | ±2 SD | ms per env plan | ±2 SD | seeds |', '|---|---|---|---|---|---|---|']
    for arm, points in amortized.items():
        for batch, s in points.items():
            call, env = s['ms_per_call'], s['ms_per_env']
            lines.append(f"| {TITLES[arm]} | {batch} | {call['mean']:.1f} | {call['sd2']:.1f} | {env['mean']:.1f} "
                         f"| {env['sd2']:.1f} | {env['n']} |")
    return lines


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    e2e, amortized = e2e_stats(), amortized_stats()
    lines = table(e2e, amortized)
    (OUT / 'beam_latency.md').write_text('\n'.join(lines) + '\n')
    plot_e2e(e2e)
    plot_amortized(amortized)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
