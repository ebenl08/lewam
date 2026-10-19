"""Wall-clock eval time of the latency runs (the eval scripts' own time.time() around world.evaluate), next to the
CUDA-event timer. Usage: python analysis/planner_comparison/walltime.py [--out <file.md>]

LeWAM: results.txt `Eval time` (scripts/eval_lewam.py). Ours: pusht_results.txt `Eval time` (PG src/eval.py) or
`evaluation_time` (PG src/leflow/eval.py); both also cover PG's per-env video recording (on in these runs). Plans per run = every plan, the warmup replan included.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from latency_compare import LATENCY, OUT
from latency_sweeps import ARMS, SWEEPS, call_stats
from plot_style import INK_2, color, style

TITLES = {**ARMS, 'grid_wH': 'LeWAM grid w=H', 'icem_faithful': 'iCEM faithful 300×10',
          'leflow_icem': 'LeFlow → iCEM 300×5', 'cem_300x30': 'CEM 300×30'}


def wall_seconds(run):
    """Last eval time in the run's results file: `Eval time: <s>s` (LeWAM, PG eval.py) or `evaluation_time: <s>`
    (PG leflow/eval.py); results files are appended per run."""
    for name in ('results.txt', 'pusht_results.txt'):
        if (run / name).exists():
            matches = re.findall(r'(?:Eval time|evaluation_time): ([0-9.]+)', (run / name).read_text())
            if matches:
                return float(matches[-1])
    return None


def total_plans(timing, config):
    if 'run' in timing:  # LeWAM timer with run totals: exact
        return timing['run']['plans']
    return timing['plans_kept'] + int(config['eval']['num_eval'])  # + the warmup replan, all envs active


def run_row(run, batch):
    timing = json.loads((run / 'timing.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    wall = wall_seconds(run)
    if wall is None:
        return None
    plans = total_plans(timing, config)
    return dict(wall_s=wall, plans=plans, episodes=int(config['eval']['num_eval']),
                wall_ms_per_plan=1000 * wall / plans, timer_ms_per_plan=call_stats(timing, batch)['ms_per_env'])


def collect(pattern, batch):
    rows = [run_row(Path(p).parent, batch) for p in sorted(Path('/').glob(pattern.lstrip('/')))]
    return [r for r in rows if r]


def mean_row(rows):
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out['n'] = len(rows)
    return out


def table(title, key_name, groups):
    lines = [f'## {title}', '', f'| Arm | {key_name} | eval wall s | plans per run | wall ms per plan | '
             'timer GPU ms per plan | wall / timer | seeds |', '|---|---|---|---|---|---|---|---|']
    for (arm, key), rows in groups.items():
        if not rows:
            continue
        m = mean_row(rows)
        lines.append(f"| {TITLES.get(arm, arm)} | {key} | {m['wall_s']:.1f} | {m['plans']:.0f} | "
                     f"{m['wall_ms_per_plan']:.1f} | {m['timer_ms_per_plan']:.1f} | "
                     f"{m['wall_ms_per_plan'] / m['timer_ms_per_plan']:.2f} | {m['n']} |")
    return lines + ['']


MARKERS = {'cem': 's', 'cem_300x30': 's', 'icem_faithful': 'v', 'leflow_cem': 'D', 'grid_w1': 'o', 'native': '^'}
HOLLOW = {'cem_300x30'}  # shares CEM's colour (a third orange shade fails the palette check): hollow markers instead


def curve(ax, arm, groups, log_x=False):
    """Wall ms per plan (solid, labelled) and the CUDA-event timer (thin dotted) over the group keys of one arm."""
    keys = [k for (a, k), rows in groups.items() if a == arm and rows]
    if not keys:
        return
    means = [mean_row(groups[(arm, k)]) for k in keys]
    kwargs = dict(color=color(arm), marker=MARKERS[arm])
    if arm in HOLLOW:
        kwargs['markerfacecolor'] = 'none'
    ax.plot(keys, [m['wall_ms_per_plan'] for m in means], linestyle='-', label=TITLES.get(arm, arm), **kwargs)
    ax.plot(keys, [m['timer_ms_per_plan'] for m in means], linestyle=':', linewidth=1, markersize=3, alpha=0.7,
            **kwargs)
    if log_x:
        ax.set_xscale('log', base=2)


def plot(horizon, batch, samples):
    """PE9: eval wall ms per plan over h (batch 1), over envs per call B (h5) and over LeWAM proposals N (h5)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for arm in ('cem', 'cem_300x30', 'icem_faithful', 'leflow_cem', 'grid_w1', 'native'):
        curve(axes[0], arm, horizon)
    axes[0].set_xticks([1, 2, 3, 4, 5, 6, 8, 10])
    axes[0].set_xlabel('horizon h (batch 1)')
    axes[0].set_title('Over horizon', fontsize=10)
    for arm in ('cem', 'leflow_cem', 'grid_w1', 'native'):
        curve(axes[1], arm, batch, log_x=True)
    axes[1].set_yscale('log')
    axes[1].set_xticks([1, 2, 4, 8, 16, 32, 64], ['1', '2', '4', '8', '16', '32', '64'])
    axes[1].set_xlabel('envs per plan call B (h = 5)')
    axes[1].set_title('Over batch (per env, amortized)', fontsize=10)
    for arm in ('native', 'grid_w1'):
        curve(axes[2], arm, samples, log_x=True)
    axes[2].set_xticks([8, 16, 32, 64, 128], ['8', '16', '32', '64', '128'])
    axes[2].set_ylim(0, None)
    axes[2].set_xlabel('LeWAM proposals N (h = 5, batch 1)')
    axes[2].set_title('Over proposals', fontsize=10)
    axes[0].set_ylabel('ms per plan')
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=INK_2, linestyle='-'), Line2D([], [], color=INK_2, linestyle=':', linewidth=1)]
    labels += ['eval wall time / plans (solid)', 'CUDA-event timer (dotted)']
    fig.legend(handles, labels, loc='outside right upper', fontsize=8)
    fig.suptitle('PushT: eval wall time per plan (time.time() around world.evaluate, mean over seeds)', fontsize=11)
    fig.text(0.01, -0.02, 'Wall time covers env stepping, rendering, encoding and Python, divided by every plan '
             '(warmup included). PG runs (CEM, iCEM, LeFlow) also record a video per env; LeWAM runs do not. '
             'Our CUDA-event timer was attached in all runs (one event pair per plan).', fontsize=7, color=INK_2,
             ha='left', va='top')
    fig.savefig(OUT / 'pe9_walltime.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    style()
    horizon = defaultdict(list)
    for arm in ('cem', 'icem_faithful', 'leflow_cem', 'leflow_icem', 'cem_300x30', 'grid_w1', 'grid_wH', 'native'):
        for h in (1, 2, 3, 4, 5, 6, 8, 10):
            if h == 1 and arm in ('grid_w1', 'grid_wH', 'native'):  # rollout_steps <= 1 means h0 in LeWAMPlanner
                continue
            root = LATENCY / 'pusht' if h in (2, 5, 10) else SWEEPS / 'horizon'
            horizon[(arm, h)] = collect(f'{root}/{arm}/h{h}/*/timing.json', 1)
    batch = defaultdict(list)
    for arm in ('cem', 'leflow_cem', 'grid_w1', 'native'):
        batch[(arm, 1)] = collect(f'{LATENCY}/pusht/{arm}/h5/*/timing.json', 1)
        for b in (2, 4, 8, 16, 32, 64):
            sweep = 'batch_rerun' if (SWEEPS / 'batch_rerun' / arm / f'h5_b{b}').exists() else 'batch'
            batch[(arm, b)] = collect(f'{SWEEPS}/{sweep}/{arm}/h5_b{b}/*/timing.json', b)
    samples = defaultdict(list)
    for arm in ('native', 'grid_w1'):
        for n in (8, 16, 32, 64, 128):
            pattern = f'{LATENCY}/pusht/{arm}/h5/*/timing.json' if n == 32 else \
                f'{SWEEPS}/samples/{arm}/h5_n{n}/*/timing.json'
            samples[(arm, n)] = collect(pattern, 1)
    lines = ['# Eval wall time of the PushT latency runs', '',
             'eval wall s = time.time() around world.evaluate in each eval script (mean over seeds); wall ms per plan '
             '= wall / every plan of the run (warmup included); timer GPU ms per plan = CUDA-event timer (warmup '
             'dropped). Wall includes env stepping, rendering, encoding and Python. PG runs (CEM, iCEM, LeFlow) also '
             'record a video per env inside world.evaluate; LeWAM runs do not (eval.save_video=false).', '']
    lines += table('Batch 1 (one env per plan call, 10 envs), over h', 'h', horizon)
    lines += table('Envs per plan call B (num_eval = B), h = 5', 'B', batch)
    lines += table('LeWAM proposals N, h = 5, batch 1', 'N', samples)
    text = '\n'.join(lines)
    print(text)
    plot(horizon, batch, samples)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
