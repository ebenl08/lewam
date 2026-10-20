"""P2c Part B: roofline and throughput over batch size from the capture-and-replay benchmark.

Reads data/planner_comparison/p2/bench/<arm>/time.json (latency, FLOPs, bytes, memory per batch) and, when present,
<arm>/nsys_bench_<arm>.sqlite (kernel time inside each "measure B=..." NVTX range). LeWAM native is averaged over
its blocks-left levels with the replan mix of the P2 native h5 runs. Writes pe12_roofline.png, pe13_throughput.png
and roofline.md to analysis/p2/. Usage: python roofline.py
"""

import json
import re
import sqlite3
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from kernel_breakdown import load_kernels, load_ranges
from latency_compare import DATA, LATENCY, OUT
from latency_sweeps import MARKERS
from plot_style import INK_2, MUTED, color, style

BENCH = DATA / 'p2' / 'bench'
ARMS = {'cem': 'LeWM CEM 300×10', 'leflow_cem': 'LeFlow → CEM 300×5', 'grid_w1': 'LeWAM constant',
        'native': 'LeWAM native'}
SPEC = dict(tflops=91.6, gbps=864.0)  # L40S datasheet: FP32 (non-tensor) and GDDR6 bandwidth
RANGE = re.compile(r'measure B=(\d+)(?: s=(\d+))?')
LABEL_OFFSETS = {'cem': (6, 4), 'leflow_cem': (6, -10), 'grid_w1': (-6, 4), 'native': (-6, -10)}  # points


def native_level_weights(max_blocks=5):
    """Share of replans at each blocks-left value: replan k of an episode has max(5 - k, 1) blocks left; mean
    replans per episode (every plan, first replan included / envs) from the P2b native h5 runs, the P2 runs whose
    timing.json holds run totals."""
    runs = sorted((DATA / 'p2' / 'breakdown_detail' / 'pusht' / 'planner').glob('native/h5/*/timing.json'))
    per_episode = np.mean([json.loads(p.read_text())['run']['plans'] /
                           json.loads((p.parent / 'config.json').read_text())['eval']['num_eval'] for p in runs])
    weights = defaultdict(float)
    k = 0
    while per_episode > 0:
        weights[max(max_blocks - k, 1)] += min(1.0, per_episode)
        per_episode -= 1
        k += 1
    total = sum(weights.values())
    return {level: w / total for level, w in weights.items()}, len(runs)


def kernel_ms_per_call(arm):
    """Mean kernel ms per timed call for each (batch, steps) range of the nsys pass, or {} without one.

    Ranges ending after the last recorded CUDA API call are skipped: the bench exits with os._exit, so the final
    CUPTI buffer of the M3 runs was never flushed (NVTX ranges outlive the CUDA trace).
    """
    db_path = BENCH / arm / f'nsys_bench_{arm}.sqlite'
    if not db_path.exists():
        return {}
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
        kernels, ranges = load_kernels(db), load_ranges(db)
        trace_end = db.execute('SELECT max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()[0]
    launch = kernels['launch'].to_numpy()
    duration = kernels['dur_ns'].to_numpy() / 1e6
    totals = defaultdict(list)
    for start, end, name in ranges[['start', 'end', 'name']].itertuples(index=False):
        match = RANGE.match(name)
        if match and end <= trace_end:
            inside = (launch >= start) & (launch <= end)
            totals[(int(match.group(1)), int(match.group(2) or 0))].append(duration[inside].sum())
    return {key: float(np.mean(v)) for key, v in totals.items()}


def load_points(arm, weights):
    """Per batch: latency, kernel time, FLOPs, bytes, memory per call; native combined over its levels."""
    path = BENCH / arm / 'time.json'
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    kernel = kernel_ms_per_call(arm)
    by_batch = defaultdict(dict)
    oom_batch = None
    for p in data['points']:
        steps = p.get('steps_left', 0) if arm == 'native' else 0
        if p.get('oom'):
            oom_batch = p['batch'] if oom_batch is None else min(oom_batch, p['batch'])
            continue
        row = dict(ms=float(np.mean(p['ms_per_call'])), gflop=p['gflop'], gbyte=p['gbyte'],
                   peak_mib=p['peak_mib'], working_mib=p['working_mib'],
                   kernel_ms=kernel.get((p['batch'], p.get('steps_left', 0) if arm in ('native', 'grid_w1') else 0)))
        by_batch[p['batch']][steps] = row
    points = {}
    for batch, levels in sorted(by_batch.items()):
        if arm == 'native':
            if set(weights) - set(levels):
                continue  # a level ran out of memory at this batch
            mix = {s: weights[s] for s in weights}
        else:
            mix = {0: 1.0}
        combined = {k: sum(w * levels[s][k] for s, w in mix.items()) for k in ('ms', 'gflop', 'gbyte')}
        combined['peak_mib'] = max(levels[s]['peak_mib'] for s in mix)
        kernels = [levels[s]['kernel_ms'] for s in mix]
        combined['kernel_ms'] = sum(w * k for (s, w), k in zip(mix.items(), kernels)) if None not in kernels else None
        points[batch] = derived(batch, combined)
    return points, dict(oom_batch=oom_batch, ceilings=data.get('ceilings'))


def derived(batch, c):
    out = dict(c)
    out['plans_per_s'] = 1000 * batch / c['ms']
    out['gflop_per_plan'] = c['gflop'] / batch
    out['gbyte_per_plan'] = c['gbyte'] / batch
    out['intensity'] = c['gflop'] / c['gbyte']
    out['wall_tflops'] = c['gflop'] / c['ms']
    out['kernel_tflops'] = c['gflop'] / c['kernel_ms'] if c['kernel_ms'] else None
    out['busy'] = 100 * c['kernel_ms'] / c['ms'] if c['kernel_ms'] else None
    return out


def ceiling_plans_per_s(point, tflops):
    """Env plans per second at a compute roof. No memory term: op-level bytes overstate DRAM traffic (achieved
    throughput already exceeds the byte-based bound for LeWAM), so they cannot serve as a hard limit."""
    return tflops * 1e3 / point['gflop_per_plan']


def practical_tflops(roof):
    """Measured large FP32 GEMM rate when available, else the datasheet FP32 rate."""
    return roof['measured']['gemm_tflops'] if roof.get('measured') else roof['tflops']


ROOFLINE_VIEWS = {  # view -> file, title, footnote
    'both': ('pe12_roofline.png', 'kernel time and wall latency',
             'Filled, solid = FLOPs / kernel time (nsys); hollow, dotted = FLOPs / wall latency per call (CUDA events, '
             'no profiler). The vertical gap is GPU idle time.'),
    'wall': ('pe12b_roofline_wall.png', 'wall latency only',
             'Points = FLOPs / wall latency per call (CUDA events, no profiler): what the planner achieves end to end, '
             'GPU idle time included.'),
    'kernel': ('pe12c_roofline_kernel.png', 'kernel time only',
               'Points = FLOPs / kernel time per call (nsys): how efficient the kernels are while they run, GPU idle '
               'time excluded.'),
}


def plot_roofline(results, roof, view='both'):
    """PE12 (both), PE12b (wall only), PE12c (kernel only): log-log roofline, one path per arm over B."""
    filename, subtitle, footnote = ROOFLINE_VIEWS[view]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    intensity = np.logspace(-1, 4, 200)
    ax.plot(intensity, np.minimum(roof['tflops'], roof['gbps'] * intensity / 1e3), color=INK_2, linewidth=1.5)
    ax.text(intensity[-1], roof['tflops'] * 1.08, f"FP32 {roof['tflops']:.1f} TFLOP/s (spec)", ha='right', fontsize=8,
            color=INK_2)
    if roof.get('measured'):
        measured = roof['measured']
        ax.plot(intensity, np.minimum(measured['gemm_tflops'], measured['copy_gbps'] * intensity / 1e3), color=MUTED,
                linewidth=1, linestyle='--')
        ax.text(intensity[-1], measured['gemm_tflops'] * 0.85, f"measured: FP32 GEMM {measured['gemm_tflops']:.1f} "
                f"TFLOP/s, copy {measured['copy_gbps']:.0f} GB/s", ha='right', fontsize=8, color=MUTED)
        if 'l2_copy_gbps' in measured:
            l2_roof = np.minimum(measured['gemm_tflops'], measured['l2_copy_gbps'] * intensity / 1e3)
            ax.plot(intensity, l2_roof, color=MUTED, linewidth=1, linestyle='-.')
            ax.text(0.12, measured['l2_copy_gbps'] * 0.12 / 1e3 * 1.3, f"L2 {measured['l2_copy_gbps']:.0f} GB/s "
                    '(measured)', fontsize=8, color=MUTED, rotation=33)
    ax.text(0.12, roof['gbps'] * 0.12 / 1e3 * 1.3, f"DRAM {roof['gbps']:.0f} GB/s", fontsize=8, color=INK_2,
            rotation=33)
    for arm, (points, _) in results.items():
        batches = list(points)
        x = [points[b]['intensity'] for b in batches]
        kernel = [points[b]['kernel_tflops'] for b in batches]
        wall = [points[b]['wall_tflops'] for b in batches]
        measured = [(b, xi, ki, wi) for b, xi, ki, wi in zip(batches, x, kernel, wall) if ki is not None]  # nsys B
        if view in ('both', 'wall'):
            ax.plot(x, wall, color=color(arm), marker=MARKERS[arm], markerfacecolor='none', linestyle=':',
                    linewidth=1, label=f'{ARMS[arm]} (wall)' if view == 'both' else ARMS[arm])
        if view in ('both', 'kernel') and measured:
            ax.plot([m[1] for m in measured], [m[2] for m in measured], color=color(arm), marker=MARKERS[arm],
                    linestyle='-', label=f'{ARMS[arm]} (kernel)' if view == 'both' else ARMS[arm])
        if view == 'both':
            for _, xi, ki, wi in measured:
                ax.plot([xi, xi], [wi, ki], color=color(arm), linewidth=0.6, alpha=0.5)
        if view == 'wall':
            labels = list(zip(batches, x, wall))
        elif view == 'kernel':
            labels = [(b, xi, ki) for b, xi, ki, _ in measured]
        else:
            labels = [(b, xi, ki if ki is not None else wi) for b, xi, ki, wi in zip(batches, x, kernel, wall)]
        for b, xi, yi in labels[:1] + labels[-1:]:  # smallest and largest B only: the middle points overlap
            ax.annotate(f'B{b}', (xi, yi), textcoords='offset points', xytext=LABEL_OFFSETS[arm], fontsize=7,
                        color=INK_2, ha='right' if LABEL_OFFSETS[arm][0] < 0 else 'left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('arithmetic intensity (FLOP per op-level byte ≈ per byte of L2 traffic)')
    ax.set_ylabel('achieved TFLOP/s')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f'PushT h5 planners on an L40S roofline, {subtitle} (labels = envs per call B)', fontsize=10)
    note = (f'{footnote} Bytes = sum of op inputs and outputs (no reuse between ops), i.e. traffic through L2; DRAM '
            'traffic is lower when tensors stay in the 96 MiB L2, so the DRAM roof can be exceeded. Roofs: L40S '
            'datasheet (solid), measured DRAM copy (dashed), measured L2-resident copy (dash-dot).')
    fig.text(0.01, -0.02, textwrap.fill(note, 170), fontsize=7, color=INK_2, ha='left', va='top')
    fig.savefig(OUT / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_throughput(results, roof):
    """PE13: env plans per second over B with each arm's roof ceiling, and GPU busy % over B."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for arm, (points, meta) in results.items():
        batches = list(points)
        rate = [points[b]['plans_per_s'] for b in batches]
        axes[0].plot(batches, rate, color=color(arm), marker=MARKERS[arm], label=ARMS[arm])
        ceiling = ceiling_plans_per_s(points[batches[-1]], practical_tflops(roof))
        axes[0].axhline(ceiling, color=color(arm), linestyle='--', linewidth=1)
        if meta['oom_batch']:
            axes[0].plot(meta['oom_batch'], rate[-1], marker='x', color=color(arm), markersize=9, linestyle='none')
        busy = [(b, points[b]['busy']) for b in batches if points[b]['busy'] is not None]
        if busy:
            axes[1].plot(*zip(*busy), color=color(arm), marker=MARKERS[arm], label=ARMS[arm])
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('envs per plan call B (fixed, replayed)')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('env plans per second')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('GPU busy % (kernel time / latency)')
    axes[1].set_ylim(0, 110)
    fig.suptitle('PushT h5: planner throughput over batch size (dashed = roofline ceiling at the largest B; '
                 '× = out of memory)', fontsize=10)
    fig.savefig(OUT / 'pe13_throughput.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def markdown(results, roof, weights, n_runs):
    lines = ['# P2c Part B: roofline and throughput over batch size (PushT h5, capture-and-replay)', '',
             f"Roofs: FP32 {roof['tflops']:.1f} TFLOP/s, DRAM {roof['gbps']:.0f} GB/s (L40S datasheet). "
             f"Measured on this GPU: {roof.get('measured', 'n/a')}. Bytes are op-level (sum of op inputs and outputs, "
             'no reuse between ops), which approximates L2 traffic, not DRAM traffic. LeWAM native is averaged over blocks-left levels with weights '
             f"{ {k: round(float(v), 3) for k, v in sorted(weights.items(), reverse=True)} } (P2b native h5, "
             f"{n_runs} runs). Ceiling = env plans per second at the measured FP32 GEMM rate (compute only).",
             '', '| Arm | B | ms per call | env plans/s | GFLOP per plan | GB per plan | FLOP/byte | wall TFLOP/s | '
             'kernel TFLOP/s | busy % | peak MiB | ceiling plans/s |', '|' + '---|' * 12]
    fmt = lambda v, f: '—' if v is None else format(v, f)
    for arm, (points, meta) in results.items():
        for b, p in points.items():
            lines.append(f"| {ARMS[arm]} | {b} | {p['ms']:.1f} | {p['plans_per_s']:.1f} | {p['gflop_per_plan']:.2f} | "
                         f"{p['gbyte_per_plan']:.3f} | {p['intensity']:.1f} | {p['wall_tflops']:.2f} | "
                         f"{fmt(p['kernel_tflops'], '.2f')} | {fmt(p['busy'], '.0f')} | {p['peak_mib']:.0f} | "
                         f"{ceiling_plans_per_s(p, practical_tflops(roof)):.0f} |")
        if meta['oom_batch']:
            lines.append(f"| {ARMS[arm]} | {meta['oom_batch']} | out of memory | | | | | | | | | |")
    return '\n'.join(lines) + '\n'


def main():
    style()
    weights, n_runs = native_level_weights()
    results = {}
    measured = None
    for arm in ARMS:
        points, meta = load_points(arm, weights)
        if points:
            results[arm] = (points, meta)
            measured = measured or meta['ceilings']
    roof = dict(SPEC, measured=measured)
    (OUT / 'roofline.md').write_text(markdown(results, roof, weights, n_runs))
    for view in ROOFLINE_VIEWS:
        plot_roofline(results, roof, view)
    plot_throughput(results, roof)
    print((OUT / 'roofline.md').read_text())


if __name__ == '__main__':
    main()
