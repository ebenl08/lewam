"""P2 sweeps on PushT: latency over envs per plan call (batch), over horizon h, and over LeWAM proposals N.

Reads data/planner_comparison/p2/sweeps/pusht/{batch,horizon,samples}/ plus the P2 batch-1 runs
(p2/latency/pusht/<arm>/h<h>/, same 3 seeds); writes plots pe5-pe7 and sweeps.md to analysis/p2/.
"""

import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from latency_compare import LATENCY, OUT, DATA
from plot_style import INK_2, SURFACE, color, is_w1, style

SWEEPS = DATA / 'p2' / 'sweeps' / 'pusht'
SEEDS = (10247, 15783, 56783)
ARMS = {'cem': 'CEM 300×10', 'leflow_cem': 'LeFlow → CEM 300×5', 'grid_w1': 'LeWAM constant (grid w=1)',
        'native': 'LeWAM native (adaptive)'}
MARKERS = {'cem': 's', 'leflow_cem': 'D', 'grid_w1': 'o', 'native': '^'}


def call_stats(timing, batch):
    """GPU ms per env plan, ms per plan call and mean envs per call over the kept replans / solves."""
    if 'replans' in timing:  # LeWAM ReplanTimer
        rows = timing['replans']
        plans = sum(r['plans'] for r in rows)
        total = sum(r['propose_ms'] + r['encode_ms'] for r in rows)
        calls = len(rows)
    else:  # stable-worldmodel timer: one kept solve (flush) = one planning step over the active envs. The `gpu`
        # stage totals are IQR-filtered; plan_per_env is the unfiltered GPU time of the kept solves per env plan
        plans, calls = timing['plans_kept'], timing['flushes_kept']
        total = timing['aggregate']['plan_per_env'] * plans
    per_env = total / plans
    if batch == 1:  # batch 1: one call per env plan
        return dict(ms_per_env=per_env, ms_per_call=per_env, envs_per_call=1.0)
    return dict(ms_per_env=per_env, ms_per_call=total / calls, envs_per_call=plans / calls)


def leflow_only_stats(timing, batch):
    """call_stats of the LeFlow part of a LeFlow → CEM run: encode + LeFlow pool, CEM refine left out.

    Encode is not its own aggregate row: plan_per_env - leflow_pool - icem_refine - unattributed."""
    agg = timing['aggregate']
    encode = agg['plan_per_env'] - agg['leflow_pool'] - agg['icem_refine'] - agg['unattributed']
    per_env = encode + agg['leflow_pool']
    if batch == 1:
        return dict(ms_per_env=per_env, ms_per_call=per_env, envs_per_call=1.0)
    plans, calls = timing['plans_kept'], timing['flushes_kept']
    return dict(ms_per_env=per_env, ms_per_call=per_env * plans / calls, envs_per_call=plans / calls)


def success_rate(run):
    for path in run.glob('*results.txt'):
        match = re.search(r"'success_rate': ([0-9.]+)", path.read_text())
        if match:
            return float(match.group(1))
    return np.nan


def load(arm, run_glob, batch=1, stats_fn=call_stats):
    """Mean over the 3 seeds of stats_fn + SR, or None when no run exists."""
    runs = [Path(p).parent for p in glob.glob(run_glob) if int(Path(p).parent.name) in SEEDS]
    if not runs:
        return None
    stats = [dict(stats_fn(json.loads((r / 'timing.json').read_text()), batch), sr=success_rate(r)) for r in runs]
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out['seed_sd_ms_per_env'] = float(np.std([s['ms_per_env'] for s in stats]))
    out['seeds'] = len(stats)
    return out


def batch_points(arm, stats_fn=call_stats):
    points = {1: load(arm, f'{LATENCY}/pusht/{arm}/h5/*/timing.json', stats_fn=stats_fn)}
    for b in (2, 4, 8, 16, 32, 64):
        # B = 32 first runs (12:26-12:31) overlapped another user's server start on the same GPUs: rerun replaces them
        sweep = 'batch_rerun' if (SWEEPS / 'batch_rerun' / arm / f'h5_b{b}').exists() else 'batch'
        points[b] = load(arm, f'{SWEEPS}/{sweep}/{arm}/h5_b{b}/*/timing.json', batch=b, stats_fn=stats_fn)
    return {k: v for k, v in points.items() if v}


def horizon_points(arm):
    points = {}
    for h in (1, 2, 3, 4, 5, 6, 8, 10):
        if h == 1 and arm in ('grid_w1', 'native'):  # LeWAMPlanner reads rollout_steps <= 1 as "h0" (= 5 on PushT)
            continue
        root = LATENCY / 'pusht' if h in (2, 5, 10) else SWEEPS / 'horizon'
        points[h] = load(arm, f'{root}/{arm}/h{h}/*/timing.json')
    return {k: v for k, v in points.items() if v}


def samples_points(arm):
    points = {32: load(arm, f'{LATENCY}/pusht/{arm}/h5/*/timing.json')}
    for n in (8, 16, 64, 128):
        points[n] = load(arm, f'{SWEEPS}/samples/{arm}/h5_n{n}/*/timing.json')
    return dict(sorted((k, v) for k, v in points.items() if v))


def line(ax, arm, xs, ys, **kwargs):
    ax.plot(xs, ys, color=color(arm), marker=MARKERS[arm], linestyle='--' if is_w1(arm) else '-',
            label=ARMS[arm], **kwargs)


def table(title, key_name, data, columns):
    lines = [f'## {title}', '', f'| Arm | {key_name} | ' + ' | '.join(c for c, _ in columns) + ' | seeds |',
             '|' + '---|' * (len(columns) + 3)]
    for arm, points in data.items():
        for key, v in points.items():
            cells = ' | '.join(fmt.format(v[c]) for c, fmt in columns)
            lines.append(f"| {ARMS[arm]} | {key} | {cells} | {v['seeds']} |")
    return lines


def plot_batch(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arm, points in data.items():
        bs = list(points)
        line(axes[0], arm, bs, [points[b]['ms_per_call'] for b in bs])
        line(axes[1], arm, bs, [points[b]['ms_per_env'] for b in bs])
    for ax, label in zip(axes, ('GPU ms per plan call', 'GPU ms per env plan (amortized)')):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ['1', '2', '4', '8', '16', '32', '64'])
        ax.set_xlabel('envs per plan call (B), h = 5')
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle('PushT: latency over batch size (mean over 3 seeds; envs that finish drop out of later calls)',
                 fontsize=10)
    fig.savefig(OUT / 'pe5_latency_vs_batch.png', dpi=150)
    plt.close(fig)


def plot_batch_leflow_only(data, leflow_only):
    """PE5b: pe5 plus the LeFlow part of LeFlow → CEM (same colour, hollow diamonds, dotted)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arm, points in data.items():
        bs = list(points)
        line(axes[0], arm, bs, [points[b]['ms_per_call'] for b in bs])
        line(axes[1], arm, bs, [points[b]['ms_per_env'] for b in bs])
    bs = list(leflow_only)
    for ax, key in zip(axes, ('ms_per_call', 'ms_per_env')):
        ax.plot(bs, [leflow_only[b][key] for b in bs], color=color('leflow_cem'), marker='D', markerfacecolor='none',
                linestyle=':', label='LeFlow only (encode + pool of LeFlow → CEM)')
    for ax, label in zip(axes, ('GPU ms per plan call', 'GPU ms per env plan (batch-averaged)')):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ['1', '2', '4', '8', '16', '32', '64'])
        ax.set_xlabel('envs per plan call (B), h = 5')
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle('PushT: latency over batch size, with the LeFlow part of LeFlow → CEM split out (mean over 3 seeds)',
                 fontsize=10)
    fig.savefig(OUT / 'pe5b_latency_vs_batch_leflow_only.png', dpi=150)
    plt.close(fig)


def plot_batch_leflow_only_bars(data, leflow_only):
    """PE5b as grouped bars per B; the LeFlow → CEM bar is stacked: LeFlow part (solid) + CEM refine (hatched)."""
    arms = ('cem', 'leflow_cem', 'grid_w1', 'native')
    batches = [1, 2, 4, 8, 16, 32, 64]
    width = 0.2
    x = np.arange(len(batches))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, key, label in zip(axes, ('ms_per_call', 'ms_per_env'),
                              ('GPU ms per plan call', 'GPU ms per env plan (batch-averaged)')):
        for i, arm in enumerate(arms):
            offset = x + (i - 1.5) * width
            totals = [data[arm][b][key] for b in batches]
            if arm != 'leflow_cem':
                ax.bar(offset, totals, width, color=color(arm), edgecolor=SURFACE, linewidth=1, label=ARMS[arm])
                continue
            leflow = [leflow_only[b][key] for b in batches]
            refine = [total - part for total, part in zip(totals, leflow)]
            ax.bar(offset, leflow, width, color=color(arm), edgecolor=SURFACE, linewidth=1,
                   label='LeFlow → CEM 300×5: LeFlow part (encode + pool)')
            ax.bar(offset, refine, width, bottom=leflow, color=SURFACE, edgecolor=color(arm), hatch='////',
                   linewidth=1, label='LeFlow → CEM 300×5: CEM refine')
        ax.set_xticks(x, [str(b) for b in batches])
        ax.set_xlabel('envs per plan call (B), h = 5')
        ax.set_ylabel(label)
        ax.grid(axis='x', visible=False)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('PushT: latency over batch size, LeFlow → CEM split into its LeFlow part and CEM refine '
                 '(mean over 3 seeds)', fontsize=10)
    fig.savefig(OUT / 'pe5b_latency_vs_batch_leflow_only_bars.png', dpi=150)
    plt.close(fig)


def plot_horizon(data):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for arm, points in data.items():
        hs = list(points)
        line(ax, arm, hs, [points[h]['ms_per_env'] for h in hs])
        ax.text(hs[-1] + 0.15, points[hs[-1]]['ms_per_env'], f"{points[hs[-1]]['ms_per_env']:.0f}", va='center',
                fontsize=8, color=INK_2)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10])
    ax.set_xlabel('horizon h (blocks of 5 env steps)')
    ax.set_ylabel('GPU ms per plan (batch 1)')
    ax.legend(fontsize=8)
    ax.set_title('PushT: latency over horizon (mean over 3 seeds)', fontsize=10)
    fig.savefig(OUT / 'pe6_latency_vs_horizon.png', dpi=150)
    plt.close(fig)


def plot_samples(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arm, points in data.items():
        ns = list(points)
        line(axes[0], arm, ns, [points[n]['ms_per_env'] for n in ns])
        line(axes[1], arm, ns, [points[n]['sr'] for n in ns])
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks([8, 16, 32, 64, 128], ['8', '16', '32', '64', '128'])
        ax.set_xlabel('LeWAM proposals N (h = 5, batch 1)')
    axes[0].set_ylabel('GPU ms per plan')
    axes[1].set_ylabel('success rate (%), 30 episodes (rh 1, offset 25)')
    axes[0].legend(fontsize=8)
    fig.suptitle('PushT: LeWAM latency and success over the number of proposals (mean over 3 seeds)', fontsize=10)
    fig.savefig(OUT / 'pe7_latency_vs_samples.png', dpi=150)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    batch = {arm: batch_points(arm) for arm in ARMS}
    horizon = {arm: horizon_points(arm) for arm in ARMS}
    samples = {arm: samples_points(arm) for arm in ('native', 'grid_w1')}
    leflow_only = batch_points('leflow_cem', stats_fn=leflow_only_stats)
    columns = [('ms_per_call', '{:.1f}'), ('ms_per_env', '{:.1f}'), ('envs_per_call', '{:.1f}'), ('sr', '{:.1f}')]
    lines = ['# P2 sweeps (PushT, GPU ms, mean over seeds 10247 15783 56783)', '',
             'ms_per_call = GPU ms of one plan call over the active envs; ms_per_env = amortized GPU ms per env plan; '
             'envs_per_call = mean active envs per call (finished envs drop out); sr = success rate (%) of the run.', '']
    lines += table('Batch (envs per plan call), h = 5', 'B', batch, columns) + ['']
    lines += ['## LeFlow only over batch: encode + LeFlow pool of the LeFlow → CEM runs (CEM refine left out), h = 5',
              '', '| B | ms_per_call | ms_per_env | envs_per_call | seeds |', '|---|---|---|---|---|']
    lines += [f"| {b} | {v['ms_per_call']:.1f} | {v['ms_per_env']:.1f} | {v['envs_per_call']:.1f} | {v['seeds']} |"
              for b, v in leflow_only.items()] + ['']
    lines += table('Horizon, batch 1', 'h', horizon, columns) + ['']
    lines += table('LeWAM proposals N, h = 5, batch 1', 'N', samples, columns)
    (OUT / 'sweeps.md').write_text('\n'.join(lines) + '\n')
    plot_batch(batch)
    plot_batch_leflow_only(batch, leflow_only)
    plot_batch_leflow_only_bars(batch, leflow_only)
    plot_horizon(horizon)
    plot_samples(samples)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
