"""P2 analysis: batch-1 GPU ms per plan on PushT, LeWAM gradient planner vs CEM / iCEM faithful / LeFlow-prior arms.

Reads data/planner_comparison/p2/latency/ (timing.json per run: LeWAM ReplanTimer or the stable-worldmodel timers) and
the P1b success rates for the SR-vs-latency plot. Writes CSV, markdown and plots to data/planner_comparison/analysis/p2/.
Per run: ms per plan = mean GPU time per plan over the kept replans (first replan dropped), history / goal encoding
included. Per config: median and IQR over seeds. Rounds = seed subsets run at different times (stability check).

CLI (from playground/lewam/):
    /home/eben/.conda/envs/lewam/bin/python analysis/planner_comparison/latency_compare.py
"""

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lewam_compare import load_arms, seed_rates
from plot_style import INK, INK_2, SURFACE, color, style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data' / 'planner_comparison'
LATENCY = DATA / 'p2' / 'latency'
OUT = DATA / 'analysis' / 'p2'
ARMS = ['native', 'grid_w1', 'grid_wH', 'cem', 'leflow_cem', 'icem_faithful', 'leflow_icem']  # grouped by family
PAPER_ARMS = ['native', 'grid_w1', 'grid_wH', 'cem_300x30']  # separate plot against the paper's CEM budget
PAPER_SPEEDUP_PUSHT = 19.9  # Table 6, LeWM CEM 300x30 / LeWAM, H100
TITLES = {'native': 'LeWAM native', 'grid_w1': 'LeWAM grid w=1', 'grid_wH': 'LeWAM grid w=H',
          'cem_300x30': 'CEM 300×30 (paper budget)', 'cem': 'CEM 300×10',
          'icem_faithful': 'iCEM faithful 300×10', 'leflow_cem': 'LeFlow → CEM 300×5', 'leflow_icem': 'LeFlow → iCEM 300×5'}
SR_ARM = {'native': 'native', 'grid_w1': 'grid_w1', 'grid_wH': 'grid_wH', 'cem': 'cem_w1',
          'icem_faithful': 'icem_faithful_w1', 'leflow_cem': 'leflow_cem_w1', 'leflow_icem': 'leflow_icem_w1'}
ROUNDS = {1: [10247, 15783], 2: [56783, 77596, 85661, 90789], 3: [68975, 23471, 27901, 91991]}  # launch rounds
ROUND_OF_SEED = {seed: rnd for rnd, seeds in ROUNDS.items() for seed in seeds}
PAPER_THROUGHPUT = {'Table 6 (H100), ms per plan': 20.8, 'Table 6 (H100), plans/s': 48.1}


def ms_per_plan(timing):
    """Mean GPU ms per plan of one run, and the stage shares per plan (ms)."""
    if 'end_to_end_ms' in timing:  # LeWAM ReplanTimer
        rows = timing['replans']
        plans = sum(r['plans'] for r in rows)
        encode = sum(r['encode_ms'] for r in rows) / plans
        propose = sum(r['propose_ms'] for r in rows) / plans
        return encode + propose, dict(encode=encode, propose=propose), plans
    plans = timing['plans_kept']
    stages = {name: stats['mean'] * stats['count'] / plans for name, stats in timing['gpu'].items()
              if timing['parent_of'].get(name) in (None, 'solve', 'end_to_end') and name not in ('solve', 'end_to_end')}
    return timing['aggregate']['plan_per_env'], stages, plans


def collect():
    rows = []
    for path in sorted(glob.glob(f'{LATENCY}/pusht/*/h*/*/timing.json')):
        run = Path(path).parent
        arm, h, seed = run.parts[-3], int(run.parts[-2][1:]), int(run.parts[-1])
        timing = json.loads(Path(path).read_text())
        gpu = (run / 'gpu.txt').read_text().split('\t')[0] if (run / 'gpu.txt').exists() else ''
        if arm == 'throughput':
            rows.append(dict(arm=arm, h=h, seed=seed, gpu=gpu, plans_per_s=timing['plans_per_s'],
                             replan_ms=timing['replan_propose_ms']['median'] + timing['replan_encode_ms']['median'],
                             plans=timing['plans_kept'], replans=timing['replans_kept']))
            continue
        ms, stages, plans = ms_per_plan(timing)
        rows.append(dict(arm=arm, h=h, seed=seed, gpu=gpu, ms_per_plan=ms, plans=plans,
                         round=ROUND_OF_SEED.get(seed, 0),
                         **{f'stage_{k}': v for k, v in stages.items()}))
    return rows


def calibration():
    out = []
    for path in sorted(glob.glob(f'{LATENCY}/calibration/*/*/timing.json')):
        run = Path(path).parent
        out.append((run.parts[-2][:12], int(run.parts[-1]), json.loads(Path(path).read_text())['aggregate']['plan_per_env']))
    return out


def cost_by_replan(arm, h):
    """Median single-env ms per plan by replan index (kept replans), over all seeds: shows the countdown workload."""
    by_index = {}
    for path in glob.glob(f'{LATENCY}/pusht/{arm}/h{h}/*/timing.json'):
        for index, replan in enumerate(json.loads(Path(path).read_text())['replans'], start=1):
            by_index.setdefault(index, []).extend(replan['plan_ms'])
    return {index: float(np.median(ms)) for index, ms in sorted(by_index.items()) if ms}


def config_stats(rows):
    """{(arm, h): dict(median, q25, q75, n, cv, per_round)} over seeds."""
    stats = {}
    for arm in ARMS + ['cem_300x30']:
        for h in (2, 5, 10):
            values = [(r['round'], r['ms_per_plan']) for r in rows if r['arm'] == arm and r['h'] == h and 'ms_per_plan' in r]
            if not values:
                continue
            ms = np.array([v for _, v in values])
            rounds = {}
            for rnd, v in values:
                rounds.setdefault(rnd, []).append(v)
            stats[(arm, h)] = dict(median=float(np.median(ms)), q25=float(np.percentile(ms, 25)),
                                   q75=float(np.percentile(ms, 75)), n=len(ms), cv=float(ms.std() / ms.mean()),
                                   per_round={k: float(np.median(v)) for k, v in sorted(rounds.items())})
    return stats


def plot_bars(stats):
    """P-E2: ms per plan per h, log scale, IQR over seeds."""
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    width = 0.8 / len(ARMS)
    for k, arm in enumerate(ARMS):
        xs, ys, low, high = [], [], [], []
        for j, h in enumerate((2, 5, 10)):
            if (arm, h) in stats:
                s = stats[(arm, h)]
                xs.append(j + k * width - 0.4 + width / 2)
                ys.append(s['median'])
                low.append(s['median'] - s['q25'])
                high.append(s['q75'] - s['median'])
        ax.bar(xs, ys, width, yerr=[low, high], capsize=2, label=TITLES[arm], color=color(arm), edgecolor=SURFACE,
               linewidth=1, error_kw={'ecolor': INK_2})
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.05, f'{y:.0f}', ha='center', va='bottom', fontsize=7, color=INK_2)
    ax.grid(axis='x', visible=False)
    ax.set_yscale('log')
    ax.set_xticks(range(3), ['h = 2', 'h = 5', 'h = 10'])
    ax.set_ylabel('GPU ms per plan (batch 1, median over seeds)')
    ax.legend(fontsize=7, ncol=4)
    fig.savefig(OUT / 'pe2_latency_bars.png', dpi=150)
    plt.close(fig)


STAGE_COLORS = {  # PG scripts/latency/plot_latency.py STAGE_COLOR (validated there), one colour per stage
    'encode': '#eda100', 'proposals': '#2a78d6', 'WM rollout': '#eb6834', 'objective': '#4a3aa7',
    'solver update': '#1baf7a', 'other': '#c3c2b7',
}
STAGE_NOTES = {
    'proposals': 'LeWAM: 32 imagined rollouts, policy + world model (no grad); LeFlow: flow sampling + inverse '
                 'dynamics + warm prior',
    'WM rollout': 'CEM: candidate rollouts; LeFlow: scoring + refinement rollouts; LeWAM: 6 differentiable '
                  'rollouts of the refinement (forward, cost included)',
    'solver update': 'CEM: elite refit + sampling; LeWAM: backward passes + Adam steps',
}
BREAKDOWN = DATA / 'p2' / 'breakdown' / 'pusht'  # LeWAM runs with the stage timers on


def stage_segments(timing):
    """Per-plan GPU ms per stage from one timing.json (LeWAM ReplanTimer or the stable-worldmodel timers)."""
    if 'replans' in timing:  # LeWAM
        rows = timing['replans']
        plans = sum(r['plans'] for r in rows)
        stage = lambda name: sum(r['stage_ms'].get(name, 0.0) for r in rows) / plans
        total = sum(r['propose_ms'] + r['encode_ms'] for r in rows) / plans
        parts = {'encode': sum(r['encode_ms'] for r in rows) / plans + stage('goal_encode'),
                 'proposals': stage('proposals'), 'WM rollout': stage('refine_forward')}
        parts['solver update'] = max(0.0, total - sum(parts.values()))
        return parts
    plans = timing['plans_kept']
    gpu = timing['gpu']
    stage = lambda name: gpu[name]['mean'] * gpu[name]['count'] / plans if name in gpu else 0.0
    total = timing['aggregate']['plan_per_env']
    if 'end_to_end' not in gpu:  # CEM / iCEM
        parts = {'WM rollout': stage('wm_rollout'), 'objective': stage('scoring') - stage('wm_rollout')}
        parts['solver update'] = max(0.0, total - stage('scoring'))
        return parts
    rollout = stage('wm_rollout') + stage('icem_rollout') + stage('warm_rollout')  # LeFlow -> CEM / iCEM
    parts = {'encode': stage('encode'),
             'proposals': stage('flow_sampling') + stage('flow_sampling_sdedit') + stage('inverse_dynamics')
                          + stage('warm_prior') - stage('warm_rollout'),
             'WM rollout': rollout,
             'objective': stage('scoring') - stage('wm_rollout') + stage('icem_scoring') - stage('icem_rollout'),
             'solver update': max(0.0, stage('icem_refine') - stage('icem_scoring'))}  # nested timers: ~0 for CEM
    parts['other'] = max(0.0, total - sum(parts.values()))
    return parts


def mean_segments(paths):
    segments = [stage_segments(json.loads(Path(p).read_text())) for p in paths]
    return {name: float(np.mean([s.get(name, 0.0) for s in segments])) for name in STAGE_COLORS}


def plot_breakdown():
    """PE3: stacked per-plan GPU ms by stage, per arm, at h = 5 and h = 10 (mean over seeds)."""
    arms = ['cem', 'leflow_cem', 'grid_w1']  # LeWAM at a fixed h (constant schedule), as CEM and LeFlow -> CEM
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharey=True)
    for ax, h in zip(axes, (5, 10)):
        labels, segments = [], []
        for arm in arms:
            root = BREAKDOWN if arm in ('native', 'grid_w1', 'grid_wH') else LATENCY / 'pusht'
            paths = glob.glob(f'{root}/{arm}/h{h}/*/timing.json')
            if paths:
                labels.append(TITLES[arm].replace(' ', '\n', 1))
                segments.append(mean_segments(paths))
        bottoms = np.zeros(len(labels))
        for name, ink in STAGE_COLORS.items():
            values = np.array([s[name] for s in segments])
            if not values.any():
                continue
            ax.bar(labels, values, bottom=bottoms, color=ink, width=0.6, edgecolor=SURFACE, linewidth=2,
                   label=name if h == 5 else None)
            bottoms += values
        for x, total in enumerate(bottoms):
            ax.text(x, total, f'{total:.0f}', ha='center', va='bottom', color=INK_2, fontsize=9)
        ax.set_title(f'h = {h}', fontsize=10)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='x', labelsize=8)
    axes[0].set_ylabel('GPU ms per plan (batch 1, mean over seeds)')
    fig.legend(loc='outside right upper', fontsize=8)
    notes = '\n'.join(f'{name}: {text}' for name, text in STAGE_NOTES.items())
    fig.text(0.01, -0.02, notes, fontsize=7, color=INK_2, ha='left', va='top')
    fig.suptitle('PushT: where the time goes per plan', fontsize=11)
    fig.savefig(OUT / 'pe3_stage_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


DETAIL_COLORS = {  # pe3 colours; WM rollout split in two orange shades, backprop takes the (negligible) objective slot
    'encode': '#eda100', 'proposals (policy / flow sampling)': '#2a78d6', 'WM rollout (no grad)': '#eb6834',
    'WM rollout (with grad)': '#a8400f', 'backprop': '#4a3aa7', 'solver update': '#1baf7a', 'other': '#c3c2b7',
}
DETAIL_NOTES = {
    'proposals': 'LeWAM: policy flow sampling, 8 Euler steps per imagined block; LeFlow: flow sampling + inverse '
                 'dynamics + warm prior',
    'WM rollout (no grad)': 'CEM: candidate rollouts; LeFlow: scoring + refinement rollouts; LeWAM: proposal rollouts',
    'WM rollout (with grad) / backprop': 'LeWAM refinement: 6 differentiable forwards, 5 backward passes',
    'solver update': 'CEM: elite refit + sampling; LeWAM: loss, zero_grad, grad clip, Adam step, action clamp',
    'other': 'objective (≤ 1 ms), bookkeeping, history slide, Python glue',
}
BREAKDOWN_DETAIL = DATA / 'p2' / 'breakdown_detail' / 'pusht' / 'planner'  # LeWAM runs, eval.timing_detail=planner
DETAIL_ARMS = {'cem': ('CEM\n300×10', LATENCY / 'pusht'), 'leflow_cem': ('LeFlow → CEM\n300×5', LATENCY / 'pusht'),
               'grid_w1': ('LeWAM constant\n(grid w=1)', BREAKDOWN_DETAIL),
               'native': ('LeWAM native\n(adaptive)', BREAKDOWN_DETAIL)}


def detail_segments(timing):
    """Per-plan GPU ms in DETAIL_COLORS groups: LeWAM from the planner-level stage tree, ours from stage_segments."""
    if 'tree' not in timing:
        parts = stage_segments(timing)
        return {'encode': parts.get('encode', 0.0),
                'proposals (policy / flow sampling)': parts.get('proposals', 0.0),
                'WM rollout (no grad)': parts.get('WM rollout', 0.0),
                'solver update': parts.get('solver update', 0.0),
                'other': parts.get('objective', 0.0) + parts.get('other', 0.0)}
    rows = timing['replans']
    plans = sum(r['plans'] for r in rows)
    total = sum(r['propose_ms'] + r['encode_ms'] for r in rows) / plans
    stage = lambda *paths: sum(timing['tree'].get(p, {}).get('gpu_ms_per_plan', 0.0) for p in paths)
    parts = {'encode': sum(r['encode_ms'] for r in rows) / plans + stage('goal_encode'),
             'proposals (policy / flow sampling)': stage('proposals/policy_sample'),
             'WM rollout (no grad)': stage('proposals/dynamics_step'),
             'WM rollout (with grad)': stage('refine_forward/dynamics_step'),
             'backprop': stage('backward'),
             'solver update': stage('loss', 'zero_grad', 'grad_clip', 'adam_step', 'action_clamp')}
    parts['other'] = max(0.0, total - sum(parts.values()))
    return parts


def plot_breakdown_detail():
    """PE4: stacked per-plan GPU ms by stage group: CEM, LeFlow -> CEM, LeWAM constant and native, h = 5 and 10."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True, sharey=True)
    table = ['| Arm | h | ' + ' | '.join(DETAIL_COLORS) + ' | total | seeds |', '|' + '---|' * (len(DETAIL_COLORS) + 4)]
    for ax, h in zip(axes, (5, 10)):
        labels, segments = [], []
        for arm, (label, root) in DETAIL_ARMS.items():
            paths = glob.glob(f'{root}/{arm}/h{h}/*/timing.json')
            if not paths:
                continue
            per_seed = [detail_segments(json.loads(Path(p).read_text())) for p in paths]
            mean = {name: float(np.mean([s.get(name, 0.0) for s in per_seed])) for name in DETAIL_COLORS}
            labels.append(label)
            segments.append(mean)
            total = sum(mean.values())
            table.append(f"| {label.replace(chr(10), ' ')} | {h} | "
                         + ' | '.join(f'{v:.1f} ({100 * v / total:.0f}%)' for v in mean.values())
                         + f' | {total:.1f} | {len(paths)} |')
        bottoms = np.zeros(len(labels))
        for name, ink in DETAIL_COLORS.items():
            values = np.array([s[name] for s in segments])
            ax.bar(labels, values, bottom=bottoms, color=ink, width=0.6, edgecolor=SURFACE, linewidth=2,
                   label=name if h == 5 else None)
            bottoms += values
        for x, total in enumerate(bottoms):
            ax.text(x, total, f'{total:.0f}', ha='center', va='bottom', color=INK_2, fontsize=9)
        ax.set_title(f'h = {h}', fontsize=10)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='x', labelsize=8)
    axes[0].set_ylabel('GPU ms per plan (batch 1, mean over seeds)')
    fig.legend(loc='outside right upper', fontsize=8)
    notes = '\n'.join(f'{name}: {text}' for name, text in DETAIL_NOTES.items())
    fig.text(0.01, -0.02, notes, fontsize=7, color=INK_2, ha='left', va='top')
    fig.suptitle('PushT: where the time goes per plan (LeWAM with planner-level stage timers)', fontsize=11)
    fig.savefig(OUT / 'pe4_stage_breakdown_detail.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'pe4_stage_breakdown_detail.md').write_text('\n'.join(table) + '\n')


def plot_vs_cem_300x30(stats):
    """PE2b: LeWAM arms vs CEM 300x30 (the paper's Table 6 budget), batch 1; left ms per plan, right speedup."""
    if not all((arm, h) in stats for arm in PAPER_ARMS for h in (2, 5, 10)):
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    width = 0.8 / len(PAPER_ARMS)
    for k, arm in enumerate(PAPER_ARMS):
        xs = [j + k * width - 0.4 + width / 2 for j in range(3)]
        rows = [stats[(arm, h)] for h in (2, 5, 10)]
        ys = [s['median'] for s in rows]
        errors = [[s['median'] - s['q25'] for s in rows], [s['q75'] - s['median'] for s in rows]]
        left.bar(xs, ys, width, yerr=errors, capsize=2, color=color(arm), edgecolor=SURFACE, linewidth=2,
                 error_kw={'ecolor': INK_2}, label=TITLES[arm])
        for x, y in zip(xs, ys):
            left.text(x, y * 1.04, f'{y:.0f}', ha='center', va='bottom', fontsize=8, color=INK_2)
    left.set_yscale('log')
    left.set_xticks(range(3), ['h = 2', 'h = 5', 'h = 10'])
    left.set_ylabel('GPU ms per plan (batch 1, median, IQR over 10 seeds)')
    left.set_title('ms per plan', fontsize=10)
    left.grid(axis='x', visible=False)
    left.legend(fontsize=8, loc='upper left')

    lewam = PAPER_ARMS[:-1]
    width = 0.8 / len(lewam)
    for k, arm in enumerate(lewam):
        xs = [j + k * width - 0.4 + width / 2 for j in range(3)]
        speedup = [stats[('cem_300x30', h)]['median'] / stats[(arm, h)]['median'] for h in (2, 5, 10)]
        right.bar(xs, speedup, width, color=color(arm), edgecolor=SURFACE, linewidth=2, label=TITLES[arm])
        for x, y in zip(xs, speedup):
            right.text(x, y * 1.04, f'{y:.2f}×', ha='center', va='bottom', fontsize=8, color=INK_2)
    right.axhline(1, color=INK_2, lw=1)
    right.text(0.5, 1.05, 'parity', ha='center', va='bottom', fontsize=8, color=INK_2)  # in the gap between groups
    right.axhline(PAPER_SPEEDUP_PUSHT, color=INK_2, lw=1, ls='--')
    right.text(2.45, PAPER_SPEEDUP_PUSHT * 1.04, f'paper Table 6, PushT: {PAPER_SPEEDUP_PUSHT}×', ha='right',
               va='bottom', fontsize=8, color=INK_2)
    right.set_yscale('log')
    right.set_ylim(0.2, 40)
    right.set_xticks(range(3), ['h = 2', 'h = 5', 'h = 10'])
    right.set_ylabel('speedup = CEM 300×30 ms / LeWAM ms (log)')
    right.set_title('LeWAM speedup over CEM 300×30 (> 1 = LeWAM faster)', fontsize=10)
    right.grid(axis='x', visible=False)
    fig.suptitle('PushT, batch 1, L40S: LeWAM gradient planner vs CEM 300×30 (the paper\'s CEM budget)', fontsize=11)
    fig.savefig(OUT / 'pe2b_latency_vs_cem300x30.png', dpi=150)
    plt.close(fig)


def plot_sr_vs_latency(stats):
    """P-N1: P1b SR vs ms per plan; left rh = 1, right best rh per h (ms per plan does not depend on rh).

    Our arms appear twice at the same latency: hollow = w=1 SR, filled = w=H SR (the window changes no compute).
    """
    arms_sr = load_arms()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, sharey=True)
    markers = {2: 'o', 5: 's', 10: '^'}
    points = [(arm, SR_ARM[arm], arm in ('native', 'grid_w1', 'grid_wH')) for arm in ARMS if arm in SR_ARM]  # no SR runs for 300x30
    points += [(arm, SR_ARM[arm].replace('_w1', '_wH'), True) for arm in ('cem', 'icem_faithful', 'leflow_cem', 'leflow_icem')]
    for ax, mode in zip(axes, ('rh1', 'best')):
        for arm, sr_arm, filled in points:
            colour = color(arm)
            for h in (2, 5, 10):
                if (arm, h) not in stats:
                    continue
                rates_by_cell = seed_rates(arms_sr[sr_arm])
                cells = [(h, 1)] if mode == 'rh1' else [(h, rh) for rh in (1, 2, 5, 10) if rh <= h]
                cell = max(cells, key=lambda c: np.mean(list(rates_by_cell[c].values())))
                rates = np.array(list(rates_by_cell[cell].values()))
                half = 1.96 * rates.std(ddof=1) / np.sqrt(len(rates))
                s = stats[(arm, h)]
                window = '' if arm in ('native', 'grid_w1', 'grid_wH') else (' w=H' if filled else ' w=1')
                ax.errorbar(s['median'], rates.mean(), xerr=[[s['median'] - s['q25']], [s['q75'] - s['median']]],
                            yerr=half, fmt=markers[h], color=colour, mfc=colour if filled else 'none', capsize=2,
                            label=TITLES[arm] + window if h == 5 else None)
                tag = f'h{h}' if mode == 'rh1' else f'h{cell[0]}rh{cell[1]}'
                ax.annotate(tag, (s['median'], rates.mean()), textcoords='offset points', xytext=(4, 3), fontsize=6)
        ax.set_xscale('log')
        ax.set_xlabel('GPU ms per plan (batch 1, median, IQR over 10 seeds)')
        ax.set_title('SR at rh = 1' if mode == 'rh1' else 'SR at the best rh for each h', fontsize=10)
    axes[0].set_ylabel('SR (%), P1b (25 seeds; LeFlow 15)')
    axes[1].legend(fontsize=7, loc='lower right')
    fig.savefig(OUT / 'pn1_sr_vs_latency.png', dpi=150)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith('stage_'), k))
    with open(OUT / 'latency_runs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    stats = config_stats(rows)

    lines = ['# P2 latency (PushT, batch 1, GPU ms per plan)', '', '## GPU calibration (CEM 300×10, h5)', '',
             '| GPU | seed | ms per plan |', '|---|---|---|']
    lines += [f'| {gpu} | {seed} | {ms:.1f} |' for gpu, seed, ms in calibration()]
    lines += ['', '## Per config (median over seeds, IQR, coefficient of variation, median per round)', '',
              '| arm | h | median ms | IQR | n | CV | per round | ratio to CEM same h |', '|---|---|---|---|---|---|---|---|']
    for (arm, h), s in stats.items():
        ratio = s['median'] / stats[('cem', h)]['median'] if ('cem', h) in stats else float('nan')
        rounds = ', '.join(f'r{k}: {v:.1f}' for k, v in s['per_round'].items())
        lines.append(f"| {TITLES[arm]} | {h} | {s['median']:.1f} | [{s['q25']:.1f}, {s['q75']:.1f}] | {s['n']} "
                     f"| {100 * s['cv']:.1f}% | {rounds} | {ratio:.2f} |")
    lines += ['', '## LeWAM ms per plan by replan index (median over seeds; index 1 = second replan, first dropped)', '',
              '| arm | h | ' + ' | '.join(f'r{i}' for i in range(1, 10)) + ' |', '|---|---|' + '---|' * 9]
    for arm in ('native', 'grid_w1'):
        for h in (5, 10):
            by_index = cost_by_replan(arm, h)
            lines.append(f'| {TITLES[arm]} | {h} | ' + ' | '.join(f'{by_index[i]:.0f}' if i in by_index else '—'
                                                                 for i in range(1, 10)) + ' |')
    throughput = [r for r in rows if r['arm'] == 'throughput']
    if throughput:
        lines += ['', '## Batched throughput (LeWAM native paper cell, 50 envs)', '',
                  '| seed | plans/s | median ms per replan | plans per replan |', '|---|---|---|---|']
        lines += [f"| {r['seed']} | {r['plans_per_s']:.1f} | {r['replan_ms']:.1f} | {r['plans'] / r['replans']:.1f} |"
                  for r in throughput]
        lines.append(f'Paper: {PAPER_THROUGHPUT}')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    plot_bars(stats)
    plot_vs_cem_300x30(stats)
    plot_breakdown()
    plot_breakdown_detail()
    plot_sr_vs_latency(stats)
    print('\n'.join(lines))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
