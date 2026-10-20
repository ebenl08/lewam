"""Summarise the nsys profiles of scripts/planner_comparison/profile_nsys.sh: kernels, launches, syncs and GPU busy
share per plan. Usage: python analysis/planner_comparison/profile_summary.py [--out <file.md>]

Totals cover the whole eval process (warmup replan, encoding and model load included) and are divided by every plan
made; GPU busy = kernel ms per plan / measured GPU ms per plan (CUDA-event timer, warmup dropped), an approximation.
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from latency_compare import DATA, LATENCY, OUT
from latency_sweeps import SWEEPS, call_stats, load
from plot_style import INK_2, SURFACE, color, style

PROFILE = DATA / 'p2' / 'profile' / 'pusht'
# LeFlow → CEM and LeWAM native B = 32 were profiled later, with NVTX ranges on (timer); NVTX profiles give the same
# kernel, launch and sync counts as the plain ones (checked on native B = 1 and constant B = 32)
PROFILE_LEFLOW = DATA / 'p2' / 'profile_nvtx' / 'pusht' / 'leflow_cem'
PROFILE_NATIVE_B32 = DATA / 'p2' / 'profile_nvtx' / 'pusht' / 'native' / 'h5_b32'
IDLE = '#c3c2b7'  # neutral: GPU waiting on the CPU
# untimed latency per plan (CUDA-event timer, no profiler; mean of 3 seeds): P2 runs for B = 1, batch sweep for B = 32
# (batch rerun where it exists: the first B = 32 runs of CEM and LeWAM overlapped another user's job)
UNTIMED = {'h5': LATENCY / 'pusht' / '{arm}' / 'h5', 'h5_b32': SWEEPS / '{sweep}' / '{arm}' / 'h5_b32'}
LABELS = {'grid_w1': 'LeWAM constant', 'native': 'LeWAM native', 'cem': 'CEM 300×10', 'leflow_cem': 'LeFlow → CEM 300×5'}
REFINE_RANGE = 'icem_refine'  # PG timer range around the CEM refine of LeFlow → CEM; ranges are inclusive in nsys
LAUNCH = ('cudaLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernel')
SYNC = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize')


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def api_calls(rows, names):
    return sum(int(r['Num Calls']) for r in rows if r['Name'].split('_v')[0] in names)


def batch_and_plans(run):
    """Envs per plan call and every plan made in a profiled run (warmup included)."""
    timing = json.loads((run / 'timing.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    batch = int(config.get('solver', {}).get('batch_size', 0) or 0) or (
        int(config['eval']['num_eval']) if config['eval'].get('timing') == 'batched' else 1)
    if 'run' in timing:  # LeWAM: every plan of the run
        plans = timing['run']['plans']
    else:                # stable-worldmodel: kept plans + the warmup solve (all envs active)
        plans = timing['plans_kept'] + int(config['eval']['num_eval'])
    return batch, plans


def untimed_ms(arm, tag, batch):
    """Untimed GPU ms per env plan (3 seeds) for a profiled run, or None when no untimed run exists."""
    sweep = 'batch_rerun' if (SWEEPS / 'batch_rerun' / arm / tag).exists() else 'batch'
    untimed = load(arm, f"{str(UNTIMED[tag]).format(arm=arm, sweep=sweep)}/*/timing.json", batch=batch)
    return untimed['ms_per_env'] if untimed else None


def summarise(run):
    timing = json.loads((run / 'timing.json').read_text())
    batch, plans = batch_and_plans(run)
    kernels = read(run / 'nsys_cuda_gpu_kern_sum.csv')
    api = read(run / 'nsys_cuda_api_sum.csv')
    memcpy = read(run / 'nsys_cuda_gpu_mem_time_sum.csv')
    kernel_ms = sum(int(r['Total Time (ns)']) for r in kernels) / 1e6
    kernel_count = sum(int(r['Instances']) for r in kernels)
    d2h = sum(int(r['Count']) for r in memcpy if 'Device-to-Host' in r['Operation'])
    h2d = sum(int(r['Count']) for r in memcpy if 'Host-to-Device' in r['Operation'])
    measured = call_stats(timing, batch)['ms_per_env']
    top = sorted(kernels, key=lambda r: -int(r['Total Time (ns)']))[:3]
    refine_ms = None
    if (run / 'nsys_nvtx_kern_sum.csv').exists():  # LeFlow → CEM: kernel time inside the CEM refine NVTX range
        refine_ms = sum(int(r['Total Time (ns)']) for r in read(run / 'nsys_nvtx_kern_sum.csv')
                        if r['NVTX Range'] == REFINE_RANGE) / 1e6 / plans
    return dict(batch=batch, plans=plans, ms_per_plan=measured, kernel_ms_per_plan=kernel_ms / plans,
                refine_kernel_ms_per_plan=refine_ms,
                busy=100 * kernel_ms / plans / measured, kernels_per_plan=kernel_count / plans,
                mean_kernel_us=1000 * kernel_ms / kernel_count, launches_per_plan=api_calls(api, LAUNCH) / plans,
                syncs_per_plan=api_calls(api, SYNC) / plans, d2h_per_plan=d2h / plans, h2d_per_plan=h2d / plans,
                top=[(r['Name'][:60], float(r['Time (%)'])) for r in top])


def plot(runs, order, filename):
    """PE8: per-plan GPU kernel time vs idle (untimed latency), kernels per plan, mean kernel time, syncs per plan."""
    order = [key for key in order if key in runs]
    names = {arm: LABELS[arm].replace(' 300', '\n300').replace('LeWAM ', 'LeWAM\n') for arm, _ in order}
    labels = [f"{names[arm]}\nB = {runs[(arm, tag)]['batch']}" for arm, tag in order]
    colors = [color(arm) for arm, _ in order]
    hatches = ['//' if runs[key]['batch'] > 1 else '' for key in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(max(13, 2.1 * len(order)), 8.5), constrained_layout=True)

    ax = axes[0, 0]
    kernel = np.array([runs[key]['kernel_ms_per_plan'] for key in order])
    latency = np.array([runs[key]['untimed_ms'] for key in order])
    idle = np.clip(latency - kernel, 0, None)
    # LeFlow → CEM: kernel time split into its LeFlow part (encode + pool) and CEM refine (in CEM's colour)
    refine = np.array([runs[key]['refine_kernel_ms_per_plan'] or 0.0 for key in order])
    ax.bar(x, kernel - refine, color=colors, hatch=hatches, edgecolor=SURFACE, linewidth=2)
    ax.bar(x, refine, bottom=kernel - refine, color=color('cem'), hatch=hatches, edgecolor=SURFACE, linewidth=2)
    ax.bar(x, idle, bottom=kernel, color=IDLE, edgecolor=SURFACE, linewidth=2)
    handles = [Patch(color=INK_2, label='GPU computing (kernels), bar in planner colour'),
               Patch(color=IDLE, label='GPU idle (launch / sync wait)')]
    if refine.any():
        handles[1:1] = [Patch(color=color('leflow_cem'), label='LeFlow → CEM: LeFlow part (encode + pool) kernels'),
                        Patch(color=color('cem'), label='LeFlow → CEM: CEM refine kernels')]
    ax.legend(handles=handles, fontsize=8, loc='upper left')
    for i in range(len(order)):
        busy = 100 * kernel[i] / latency[i]
        busy_text = f'{busy:.0f}% busy' if busy <= 100 else '≈100% busy*'  # * see footnote: sources differ
        ax.text(i, max(latency[i], kernel[i]), f'{latency[i]:.0f} ms\n{busy_text}', ha='center', va='bottom',
                fontsize=8, color=INK_2)
    ax.set_ylabel('GPU ms per plan (batch-averaged)')
    ax.set_ylim(0, latency.max() * 1.2)
    ax.set_title('Where one plan\'s latency goes', fontsize=10)

    panels = [(axes[0, 1], 'kernels_per_plan', 'kernels launched per plan (batch-averaged)', '{:.0f}'),
              (axes[1, 0], 'mean_kernel_us', 'mean kernel duration (µs)', '{:.1f}'),
              (axes[1, 1], 'syncs_per_plan', 'CPU-GPU stream syncs per plan (batch-averaged)', '{:.0f}')]
    for ax, key, title, fmt in panels:
        values = np.array([runs[k][key] for k in order])
        ax.bar(x, values, color=colors, hatch=hatches, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(values):
            ax.text(i, v * 1.1, fmt.format(v), ha='center', va='bottom', fontsize=8, color=INK_2)
        ax.set_yscale('log')
        ax.set_ylim(values.min() / 2, values.max() * 5)
        ax.set_title(title, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis='x', visible=False)
    fig.suptitle('PushT h5, nsys profile (seed 10247): GPU busy vs launch / sync overhead; hatched = 32 envs per call',
                 fontsize=11)
    fig.text(0.01, -0.01, 'Kernel time from nsys (not changed by the profiler); latency = untimed CUDA-event timer, mean of '
             '3 seeds (P2 for B = 1, batch sweep for B = 32, rerun where it exists); per-plan counts include the warmup '
             'replan. Batch-averaged = run total / plans made: at B = 32 one call (about 24k kernels for LeWAM) serves '
             '19-32 envs.\n* kernel time (one seed, warmup replan included) slightly exceeds the 3-seed latency, so the '
             'GPU is fully busy within measurement error.', fontsize=7, color=INK_2, ha='left', va='top')
    fig.savefig(OUT / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    style()
    runs = {}
    lines = ['Per plan = run total / plans made: at B > 1 one call serves several envs, so per-plan values are '
             'batch-averaged over the envs in a call (one call launches about the same kernels at any B).', '',
             '| Run | B | ms per plan (timer, batch-averaged) | kernel ms per plan (batch-averaged) | GPU busy % '
             '| kernels per plan (batch-averaged) | mean kernel µs | launches per plan (batch-averaged) '
             '| stream syncs per plan (batch-averaged) | D2H copies per plan (batch-averaged) '
             '| H2D copies per plan (batch-averaged) |',
             '|' + '---|' * 11]
    tops = []
    paths = sorted(glob.glob(f'{PROFILE}/*/*/*/nsys_cuda_gpu_kern_sum.csv'))
    paths += sorted(glob.glob(f'{PROFILE_LEFLOW}/*/*/nsys_cuda_gpu_kern_sum.csv'))
    paths += sorted(glob.glob(f'{PROFILE_NATIVE_B32}/*/nsys_cuda_gpu_kern_sum.csv'))
    for path in paths:
        run = Path(path).parent
        s = summarise(run)
        arm, tag = run.parts[-3], run.parts[-2]
        untimed = untimed_ms(arm, tag, s['batch'])
        s['untimed_ms'] = untimed if untimed is not None else s['ms_per_plan']
        runs[(arm, tag)] = s
        name = f"{LABELS.get(arm, arm)} {tag}"
        lines.append(f"| {name} | {s['batch']} | {s['ms_per_plan']:.1f} | {s['kernel_ms_per_plan']:.1f} | "
                     f"{s['busy']:.0f} | {s['kernels_per_plan']:.0f} | {s['mean_kernel_us']:.1f} | "
                     f"{s['launches_per_plan']:.0f} | {s['syncs_per_plan']:.1f} | {s['d2h_per_plan']:.1f} | "
                     f"{s['h2d_per_plan']:.1f} |")
        tops.append(f"- {name}: " + '; '.join(f'{n} ({p:.0f}%)' for n, p in s['top']))
    text = '\n'.join(lines + ['', 'Top kernels by GPU time:'] + tops) + '\n'
    print(text)
    order = [('cem', 'h5'), ('cem', 'h5_b32'), ('native', 'h5'), ('native', 'h5_b32'), ('grid_w1', 'h5'),
             ('grid_w1', 'h5_b32')]
    plot(runs, order, 'pe8_profile.png')
    order_b = [('cem', 'h5'), ('cem', 'h5_b32'), ('leflow_cem', 'h5'), ('leflow_cem', 'h5_b32'), ('native', 'h5'),
               ('native', 'h5_b32'), ('grid_w1', 'h5'), ('grid_w1', 'h5_b32')]
    plot(runs, order_b, 'pe8b_profile_leflow.png')
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
