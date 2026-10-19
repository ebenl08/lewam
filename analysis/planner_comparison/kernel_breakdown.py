"""P2c Part A: GPU kernel time per env plan by planner component and by kernel class, from the nsys NVTX profiles.

Each kernel is joined to the runtime call that launched it and labelled with the main-thread NVTX ranges open at
the launch time, whatever thread launched it (LeWAM backward kernels come from the autograd thread). Writes per-run
aggregates to p2/kernels/, plots pe10 / pe11 and kernel_breakdown.md. Usage: python kernel_breakdown.py [--refresh]
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from latency_compare import DATA, DETAIL_COLORS, OUT
from plot_style import INK_2, MUTED, SURFACE, style
from profile_summary import IDLE, batch_and_plans, untimed_ms

PROFILE = DATA / 'p2' / 'profile_nvtx' / 'pusht'
KERNELS = DATA / 'p2' / 'kernels'
RUNS = [('cem', 'h5'), ('cem', 'h5_b32'), ('leflow_cem', 'h5'), ('leflow_cem', 'h5_b32'),
        ('native', 'h5'), ('native', 'h5_b32'), ('grid_w1', 'h5'), ('grid_w1', 'h5_b32')]
ARM_LABELS = {'cem': 'LeWM CEM 300×10', 'leflow_cem': 'LeFlow → CEM 300×5', 'native': 'LeWAM native',
              'grid_w1': 'LeWAM constant'}

PROPOSALS = 'proposals (policy / flow sampling)'
COMPONENTS = list(DETAIL_COLORS)  # encode, proposals, WM no grad, WM with grad, backprop, solver update, other
COMPONENT_COLORS = {**DETAIL_COLORS, 'other': MUTED}

# fine kernel class: first regex match on the demangled name wins (convolution before GEMM: cuDNN names hold "gemm")
KERNEL_CLASSES = [
    ('convolution', r'fprop|convolve|cudnn|winograd|max_pool|nchwToNhwc|nhwcToNchw'),
    ('attention', r'fmha|flash_|attention'),
    ('GEMM', r'gemm|gemv|splitKreduce|cublasLt'),
    ('normalization', r'layer_norm|batch_norm|group_norm|RowwiseMoments|ComputeFusedParams'),
    ('sort / top-k', r'RadixSort|TopK|bitonicSort'),
    ('random / FFT', r'distribution_|philox|curand|fft'),
    ('index / select', r'index_elementwise|indexing_backward|gather|scatter|DeviceSelect|DeviceCompact|nonzero'),
    ('reduction', r'reduce_kernel|DeviceReduce|softmax|lpnorm'),
    ('copy / concat', r'CatArray|copy_kernel|memset'),
    ('optimizer', r'multi_tensor_apply'),
    ('elementwise', r'elementwise'),
]
CLASS_GROUPS = {  # fine class -> plotted group (8 groups, one categorical hue each, neutral last)
    'GEMM': 'GEMM', 'attention': 'attention', 'convolution': 'convolution (encoder)',
    'normalization': 'normalization', 'elementwise': 'elementwise', 'optimizer': 'elementwise',
    'reduction': 'reduction', 'index / select': 'data movement', 'sort / top-k': 'data movement',
    'copy / concat': 'data movement', 'random / FFT': 'other', 'other': 'other',
}
GROUP_COLORS = {'GEMM': '#2a78d6', 'attention': '#eb6834', 'convolution (encoder)': '#1baf7a',
                'normalization': '#eda100', 'elementwise': '#e87ba4', 'reduction': '#008300',
                'data movement': '#6250d6', 'other': IDLE}

LIBRARY_RANGE = re.compile(r'^(cub::|cuBLAS|CCCL|cublas)')
LEWAM_UPDATE = {'loss', 'zero_grad', 'grad_clip', 'adam_step', 'action_clamp', 'best_update'}
LEFLOW_PROPOSALS = {'flow_sampling', 'flow_step', 'flow_sampling_sdedit', 'flow_step_sdedit', 'inverse_dynamics',
                    'warm_prior', 'warm_rollout', 'warm_predict_step'}


def lewam_component(names):
    if 'backward' in names:
        return 'backprop'
    if 'refine_forward' in names:
        return 'other' if 'cost' in names else 'WM rollout (with grad)'
    if 'policy_sample' in names or 'noise' in names:
        return PROPOSALS
    if 'proposals' in names:
        return 'other' if 'cost' in names else 'WM rollout (no grad)'
    if names & LEWAM_UPDATE:
        return 'solver update'
    if names & {'encode_history', 'goal_encode'}:
        return 'encode'
    return 'other'


def cem_component(names):
    if names & {'wm_rollout', 'wm_predict_step'}:
        return 'WM rollout (no grad)'
    if 'scoring' in names:
        return 'other'
    return 'solver update'  # sampling, elite refit and encode: the PG CEM timer has no range for them


def leflow_component(names):
    if names & {'encode', 'encode_pixels'}:
        return 'encode'
    if names & {'icem_rollout', 'icem_predict_step', 'wm_rollout', 'wm_predict_step'}:
        return 'WM rollout (no grad)'
    if names & LEFLOW_PROPOSALS:
        return PROPOSALS
    if names & {'icem_scoring', 'scoring'}:
        return 'other'
    if 'icem_refine' in names:
        return 'solver update'
    return 'other'


COMPONENT_FN = {'cem': cem_component, 'leflow_cem': leflow_component, 'native': lewam_component,
                'grid_w1': lewam_component}
CALL_RANGE = {'cem': 'solve', 'leflow_cem': 'solve', 'native': 'replan', 'grid_w1': 'replan'}  # one plan call


def kernel_class(name):
    for label, pattern in KERNEL_CLASSES:
        if re.search(pattern, name):
            return label
    return 'other'


def load_kernels(db):
    """One row per kernel: duration, names and launch time (CPU timestamp of its launch call)."""
    return pd.read_sql_query("""
        SELECT k.end - k.start AS dur_ns, d.value AS demangled, s.value AS short, r.start AS launch
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
        JOIN StringIds d ON k.demangledName = d.id
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY r.start""", db)


def load_ranges(db):
    """Planner NVTX ranges of the main thread (the thread with the most of them), library ranges dropped."""
    ranges = pd.read_sql_query("""
        SELECT n.start, n.end, coalesce(n.text, s.value) AS name, n.globalTid AS tid
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.end IS NOT NULL""", db)
    ranges = ranges[~ranges['name'].str.match(LIBRARY_RANGE)]
    main_tid = ranges['tid'].value_counts().idxmax()
    ranges = ranges[ranges['tid'] == main_tid]
    return ranges.sort_values(['start', 'end'], ascending=[True, False])


def label_launches(launches, ranges, component_fn):
    """Component of each launch time (sorted): sweep over the nested ranges, keeping the open ones on a stack."""
    starts, ends, names = ranges['start'].to_numpy(), ranges['end'].to_numpy(), ranges['name'].to_numpy()
    stack, cache, labels, next_range = [], {}, [], 0
    for t in launches:
        while next_range < len(starts) and starts[next_range] <= t:
            while stack and ends[stack[-1]] < starts[next_range]:
                stack.pop()
            stack.append(next_range)
            next_range += 1
        while stack and ends[stack[-1]] < t:
            stack.pop()
        key = frozenset(names[i] for i in stack)
        if key not in cache:
            cache[key] = component_fn(key)
        labels.append(cache[key])
    return labels


def attribute(arm, tag, refresh=False):
    """Per-run aggregate (component, class, kernel) -> total ms and count; cached as CSV in p2/kernels/."""
    out = KERNELS / f'{arm}_{tag}.csv'
    if out.exists() and not refresh:
        return pd.read_csv(out)
    db_path = next((PROFILE / arm / tag / '10247').glob('*.sqlite'), None)
    if db_path is None:
        return None
    with sqlite3.connect(db_path) as db:
        kernels, ranges = load_kernels(db), load_ranges(db)
    kernels['component'] = label_launches(kernels['launch'].to_numpy(), ranges, COMPONENT_FN[arm])
    kernels['class'] = [kernel_class(name) for name in kernels['demangled']]
    # the timers drop the first plan call (warmup); so do we: kernels launched up to the end of the first call range
    calls = ranges[ranges['name'].str.startswith(CALL_RANGE[arm])]
    kernels['warmup'] = kernels['launch'] <= calls['end'].iloc[0]
    table = (kernels.groupby(['warmup', 'component', 'class', 'short'])
             .agg(ms=('dur_ns', lambda d: d.sum() / 1e6), count=('dur_ns', 'size')).reset_index())
    KERNELS.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def kept_plans(run):
    """Env plans of the kept (non-warmup) calls, as the timers count them."""
    timing = json.loads((run / 'timing.json').read_text())
    if 'replans' in timing:
        return sum(r['plans'] for r in timing['replans'])
    return timing['plans_kept']


def run_summary(arm, tag, table):
    run = PROFILE / arm / tag / '10247'
    batch, _ = batch_and_plans(run)
    plans = kept_plans(run)
    table = table[~table['warmup']]
    kernel_ms = table['ms'].sum() / plans
    latency = untimed_ms(arm, tag, batch)
    components = table.groupby('component')['ms'].sum() / plans
    groups = table.assign(group=table['class'].map(CLASS_GROUPS)).groupby('group')['ms'].sum() / plans
    return dict(arm=arm, tag=tag, batch=batch, plans=plans, kernel_ms=kernel_ms, latency=latency,
                idle=max(0.0, latency - kernel_ms) if latency else 0.0,
                components={c: float(components.get(c, 0.0)) for c in COMPONENTS},
                groups={g: float(groups.get(g, 0.0)) for g in GROUP_COLORS},
                classes=(table.groupby('class')['ms'].sum() / plans).to_dict(),
                kernels_per_plan=table['count'].sum() / plans,
                top=(table.groupby(['short', 'class', 'component'])['ms'].sum() / plans)
                .sort_values(ascending=False).head(10))


def row_label(s):
    return f"{ARM_LABELS[s['arm']]}, B = {s['batch']}"


def plot_components(summaries):
    """PE10: GPU ms per env plan by component plus GPU idle; B = 1 and B = 32 in separate panels (own x scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for ax, batch_one in zip(axes, (True, False)):
        rows = [s for s in summaries if (s['batch'] == 1) == batch_one][::-1]
        for y, s in enumerate(rows):
            left = 0.0
            for comp in COMPONENTS:
                width = s['components'][comp]
                ax.barh(y, width, left=left, color=COMPONENT_COLORS[comp], edgecolor=SURFACE, linewidth=1.5)
                left += width
            ax.barh(y, s['idle'], left=left, color=SURFACE, edgecolor=IDLE, hatch='////', linewidth=1)
            total = s['latency'] or left
            busy = 100 * s['kernel_ms'] / total
            busy_text = f'{busy:.0f}% busy' if busy <= 100 else '≈100% busy*'
            ax.text(max(left + s['idle'], total) * 1.01, y, f"{total:.0f} ms, {busy_text}", va='center',
                    fontsize=8, color=INK_2)
        ax.set_yticks(range(len(rows)), [row_label(s) for s in rows], fontsize=8)
        ax.set_xlim(0, max((s['latency'] or s['kernel_ms']) for s in rows) * 1.3)
        ax.set_xlabel('GPU ms per env plan (batch-averaged)')
        ax.set_title('one env per plan call' if batch_one else '32 envs per plan call', fontsize=10)
        ax.grid(axis='y', visible=False)
    handles = [Patch(color=COMPONENT_COLORS[c], label=c) for c in COMPONENTS]
    handles.append(Patch(facecolor=SURFACE, edgecolor=IDLE, hatch='////', label='GPU idle (launch / sync wait)'))
    fig.legend(handles=handles, loc='outside right center', fontsize=8)
    fig.suptitle('PushT h5: where each plan\'s GPU time goes, by planner component (nsys, seed 10247)', fontsize=11)
    fig.text(0.01, -0.02, 'Bar = kernel time of the profiled run per env plan (one seed, warmup call dropped) + GPU idle '
             '(untimed latency, 3 seeds, minus kernel time). * kernel time slightly exceeds the 3-seed latency (one seed; '
             'LeFlow → CEM was profiled on a GPU with another job at ~13% load): GPU fully busy within measurement '
             'error.', fontsize=7, color=INK_2, ha='left', va='top')
    fig.savefig(OUT / 'pe10_kernel_components.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_classes(summaries):
    """PE11: share of kernel time by kernel class group, every run."""
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    rows = summaries[::-1]
    for y, s in enumerate(rows):
        left = 0.0
        for group, colour in GROUP_COLORS.items():
            share = 100 * s['groups'][group] / s['kernel_ms']
            ax.barh(y, share, left=left, color=colour, edgecolor=SURFACE, linewidth=1.5)
            if share >= 8:
                ax.text(left + share / 2, y, f'{share:.0f}%', ha='center', va='center', fontsize=7, color=SURFACE)
            left += share
    ax.set_yticks(range(len(rows)), [row_label(s) for s in rows], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of kernel time')
    ax.grid(axis='y', visible=False)
    fig.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLORS.items()], loc='outside lower center',
               ncol=4, fontsize=8)
    ax.set_title('PushT h5: kernel time by kernel class (nsys, seed 10247)', fontsize=11)
    fig.savefig(OUT / 'pe11_kernel_classes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


GLOSSARY = [
    ('gemmSN_TN_kernel, gemmSN_NN_kernel', 'GEMM', 'cuBLAS matrix multiply for a small N dimension (skinny output); '
     'chosen when one side of the product is narrow, e.g. weight gradients or per-row conditioning'),
    ('ampere_sgemm_*', 'GEMM', 'FP32 matrix multiply on CUDA cores (no tensor cores; TF32 matmul is off in both envs)'),
    ('splitKreduce_kernel', 'GEMM', 'second step of a split-K GEMM: sums partial products computed in parallel'),
    ('internal::gemvx::kernel', 'GEMM', 'cuBLAS matrix-vector product'),
    ('cublasLt::epilogue::impl::globalKernel', 'GEMM', 'cuBLASLt epilogue (bias add / activation after a GEMM)'),
    ('fmha_cutlassF_*, fmha_cutlassB_*', 'attention', 'memory-efficient attention (PyTorch SDPA, CUTLASS): '
     'forward (F) and backward (B)'),
    ('sm86_xmma_fprop_*tf32*, cutlass_tensorop_*fprop*, implicit_convolve_sgemm, *scudnn*', 'convolution',
     'cuDNN convolutions of the image encoders (TF32 tensor cores: cuDNN TF32 is on)'),
    ('nchwToNhwcKernel, nhwcToNchwKernel, max_pool_forward_nchw', 'convolution', 'layout transforms and pooling '
     'around the encoder convolutions'),
    ('vectorized_layer_norm_kernel, layer_norm_grad_input_kernel_vectorized', 'normalization',
     'LayerNorm forward and backward'),
    ('RowwiseMomentsCUDAKernel, ComputeFusedParamsCUDAKernel', 'normalization', 'GroupNorm statistics (LeWAM encoder)'),
    ('batch_norm_transform_input_channels_last_kernel', 'normalization', 'BatchNorm in inference (LeWM encoder)'),
    ('vectorized_elementwise_kernel, elementwise_kernel, unrolled_elementwise_kernel', 'elementwise',
     'pointwise ops: add, multiply, GELU, SiLU, AdaLN scale / shift, Euler update, clamp'),
    ('multi_tensor_apply_kernel', 'elementwise', 'fused Adam update over the optimised tensors'),
    ('reduce_kernel, DeviceReduce*, lpnorm_cleanup', 'reduction', 'sums, means and norms: costs, elite mean / std, '
     'gradient norm'),
    ('index_elementwise_kernel, indexing_backward_kernel', 'data movement', 'advanced indexing forward and backward '
     '(e.g. LeWAM token scatter with index lists)'),
    ('DeviceSelect*, DeviceCompactInit', 'data movement', 'nonzero / boolean-mask selection; the result size goes back '
     'to the CPU, so each is a sync point'),
    ('gatherTopK, bitonicSortKVInPlace, DeviceRadixSort*', 'data movement', 'top-k elite selection and sorting'),
    ('CatArrayBatchedCopy*', 'data movement', 'torch.cat (e.g. sliding the history, stacking rollout steps)'),
    ('distribution_elementwise_grid_stride_kernel', 'other', 'random number generation (randn noise)'),
    ('vector_fft_r2c, vector_fft_c2r', 'other', 'FFTs of the coloured-noise sampler in the CEM refine'),
]
COMPONENT_NOTES = {
    'encode': 'image encoder on the current and goal frames. LeWM CEM has no range for it; its encoder kernels '
              '(< 0.1 ms per plan) fall in WM rollout and other',
    PROPOSALS: 'LeWAM: flow policy, 8 Euler steps per imagined block (plus noise draws). LeFlow → CEM: flow sampling, '
               'SDEdit warm start, inverse dynamics and the warm-prior rollout',
    'WM rollout (no grad)': 'world-model predictor steps scored without gradient: CEM candidates, LeFlow pool and CEM '
                            'refine rollouts, LeWAM proposal rollouts',
    'WM rollout (with grad)': 'LeWAM refinement: 6 differentiable rollouts of 5 blocks',
    'backprop': 'LeWAM: 5 backward passes through the differentiable rollout (launched from the autograd thread)',
    'solver update': 'LeWM CEM: sampling and elite refit (kernels outside the scoring range). LeFlow → CEM: CEM '
                     'sampling and refit. LeWAM: loss, zero_grad, gradient clip, Adam, clamp, best-plan record',
    'other': 'costs / objectives, setup, selection, stats and kernels outside every planner range',
    'GPU idle': 'untimed latency per plan minus kernel time: the GPU waits for kernel launches or a CPU sync',
}


def markdown(summaries):
    lines = ['# P2c Part A: kernel distribution (PushT h5, nsys seed 10247)', '',
             'Kernel ms per env plan = kernel time of the profiled run / env plans, first plan call (warmup) dropped as '
             'the timers do; at '
             'B = 32 one call serves several envs, so values are batch-averaged. Latency = untimed CUDA-event timer, '
             '3 seeds (P2 runs for B = 1, batch sweep for B = 32). Components follow the planners\' NVTX ranges, '
             'attributed by launch time (backward kernels from the autograd thread included).', '',
             '## Kernel ms per env plan by component', '',
             '| Run | ' + ' | '.join(COMPONENTS) + ' | kernel total | GPU idle | latency | busy % |',
             '|' + '---|' * (len(COMPONENTS) + 5)]
    for s in summaries:
        cells = ' | '.join(f"{s['components'][c]:.1f}" for c in COMPONENTS)
        latency = s['latency'] or float('nan')
        lines.append(f"| {row_label(s)} | {cells} | {s['kernel_ms']:.1f} | {s['idle']:.1f} | {latency:.1f} | "
                     f"{100 * s['kernel_ms'] / latency:.0f} |")
    classes = [label for label, _ in KERNEL_CLASSES] + ['other']
    lines += ['', '## Share of kernel time by kernel class (%)', '',
              '| Run | ' + ' | '.join(classes) + ' | kernels per plan |', '|' + '---|' * (len(classes) + 2)]
    for s in summaries:
        cells = ' | '.join(f"{100 * s['classes'].get(c, 0.0) / s['kernel_ms']:.1f}" for c in classes)
        lines.append(f"| {row_label(s)} | {cells} | {s['kernels_per_plan']:.0f} |")
    lines += ['', '## Top 10 kernels per run (ms per env plan)', '']
    for s in summaries:
        lines += [f'### {row_label(s)}', '', '| Kernel | Class | Component | ms per env plan | % of kernel time |',
                  '|---|---|---|---|---|']
        for (short, cls, comp), ms in s['top'].items():
            lines.append(f"| `{short[:60]}` | {cls} | {comp} | {ms:.2f} | {100 * ms / s['kernel_ms']:.1f} |")
        lines.append('')
    lines += ['## Glossary: components', '', '| Component | What it covers |', '|---|---|']
    lines += [f'| {c} | {note} |' for c, note in COMPONENT_NOTES.items()]
    lines += ['', '## Glossary: kernels', '', '| Kernel names | Class | Meaning |', '|---|---|---|']
    lines += [f'| `{names}` | {cls} | {meaning} |' for names, cls, meaning in GLOSSARY]
    lines += ['', '## Glossary: other labels', '', '| Label | Meaning |', '|---|---|',
              '| batch-averaged | run total ÷ env plans; one call at B > 1 serves B envs |',
              '| GPU busy % | kernel time ÷ untimed latency per plan |',
              '| plotted class groups | data movement = index / select + sort / top-k + copy / concat; '
              'elementwise includes the fused optimizer; other = random / FFT + unmatched |']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='recompute the cached per-run CSVs')
    args = parser.parse_args()
    style()
    summaries = []
    for arm, tag in RUNS:
        table = attribute(arm, tag, refresh=args.refresh)
        if table is None:
            print(f'no profile for {arm} {tag}, skipped')
            continue
        summaries.append(run_summary(arm, tag, table))
    text = markdown(summaries)
    (OUT / 'kernel_breakdown.md').write_text(text)
    plot_components(summaries)
    plot_classes(summaries)
    print(text)


if __name__ == '__main__':
    main()
