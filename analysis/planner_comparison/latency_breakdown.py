"""P2b: detailed LeWAM stage breakdown tables from timing.json files written with eval.timing_detail.

Usage: python analysis/planner_comparison/latency_breakdown.py <root> [<root> ...] [--out <file.md>]
<root> holds <arm>/h<h>/<seed>/timing.json; numbers are means over seeds.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# user-facing groups of the stage tree (paths relative to one plan)
GROUPS = {
    'goal encode': ['goal_encode'],
    'policy sampling (proposals)': ['proposals/policy_sample'],
    'WM rollout (proposals, no grad)': ['proposals/dynamics_step'],
    'WM rollout (refinement forward)': ['refine_forward/dynamics_step'],
    'backprop': ['backward'],
    'plan update (loss, zero_grad, clip, Adam, clamp)': ['loss', 'zero_grad', 'grad_clip', 'adam_step', 'action_clamp'],
    'cost (proposals + refinement)': ['proposals/cost', 'refine_forward/cost'],
    'noise draws': ['proposals/noise'],
    'bookkeeping (setup, select, best update, stats)': ['proposal_setup', 'warm_select', 'refine_setup',
                                                        'best_update', 'final_select', 'stats'],
}


def load(root):
    runs = defaultdict(list)
    for path in sorted(glob.glob(f'{root}/*/h*/*/timing.json')):
        arm, h = Path(path).parts[-4], int(Path(path).parts[-3][1:])
        runs[(arm, h)].append(json.loads(Path(path).read_text()))
    return runs


def plan_ms(timing):
    plans = [ms for row in timing['replans'] for ms in row['plan_ms']]
    return float(np.mean(plans))


def mean_tree(timings):
    """Per path: mean over seeds of calls per plan, GPU ms per call, GPU and host ms per plan."""
    paths = dict.fromkeys(p for t in timings for p in t['tree'])
    keys = ('calls_per_plan', 'gpu_ms_per_call', 'gpu_ms_per_plan', 'host_ms_per_plan')
    return {p: {k: float(np.mean([t['tree'][p][k] for t in timings if p in t['tree']])) for k in keys} for p in paths}


def with_remainders(tree, total):
    """Add `<parent>/(unattributed)` = parent − children, and top-level `(unattributed)` = plan − top-level stages."""
    out = dict(tree)
    for parent in [p for p in tree]:
        children = [p for p in tree if '/' in p and p.rsplit('/', 1)[0] == parent]
        if children:
            rest = tree[parent]['gpu_ms_per_plan'] - sum(tree[c]['gpu_ms_per_plan'] for c in children)
            out[f'{parent}/(unattributed)'] = dict(calls_per_plan=np.nan, gpu_ms_per_call=np.nan,
                                                   gpu_ms_per_plan=rest, host_ms_per_plan=np.nan)
    top = sum(v['gpu_ms_per_plan'] for p, v in tree.items() if '/' not in p)
    out['(unattributed)'] = dict(calls_per_plan=np.nan, gpu_ms_per_call=np.nan, gpu_ms_per_plan=total - top,
                                 host_ms_per_plan=np.nan)
    return out


def children_of(paths):
    """Parent path ('' for the plan) -> child paths, in first-seen order (remainders come last)."""
    children = defaultdict(list)
    for p in paths:
        children[p.rsplit('/', 1)[0] if '/' in p else ''].append(p)
    return children


def fmt(x, digits=1):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{digits}f}'


def grouped_table(runs):
    keys = sorted(runs)
    head = '| Group | ' + ' | '.join(f'{a} h{h} ms (%)' for a, h in keys) + ' |'
    lines = [head, '|' + '---|' * (len(keys) + 1)]
    trees = {k: mean_tree(runs[k]) for k in keys}
    totals = {k: float(np.mean([plan_ms(t) for t in runs[k]])) for k in keys}
    grouped = {k: {g: sum(trees[k].get(p, {}).get('gpu_ms_per_plan', 0.0) for p in paths)
                   for g, paths in GROUPS.items()} for k in keys}
    for group in GROUPS:
        lines.append(f'| {group} | ' + ' | '.join(
            f'{grouped[k][group]:.1f} ({100 * grouped[k][group] / totals[k]:.1f}%)' for k in keys) + ' |')
    rest = {k: totals[k] - sum(grouped[k].values()) for k in keys}
    lines.append('| other (glue: history slide, goal inputs, Python) | ' + ' | '.join(
        f'{rest[k]:.1f} ({100 * rest[k] / totals[k]:.1f}%)' for k in keys) + ' |')
    lines.append('| **plan total** | ' + ' | '.join(f'**{totals[k]:.1f}** (100%)' for k in keys) + ' |')
    return '\n'.join(lines)


def tree_text(timings):
    """Plain-text stage tree (box-drawing branches) with calls, GPU ms per call / per plan, % of plan, host ms."""
    total = float(np.mean([plan_ms(t) for t in timings]))
    tree = with_remainders(mean_tree(timings), total)
    children = children_of(tree)
    rows = []

    def walk(parent, prefix):
        kids = children.get(parent, [])
        for i, path in enumerate(kids):
            last = i == len(kids) - 1
            name = path.rsplit('/', 1)[-1]
            label = name if not parent else prefix + ('└─ ' if last else '├─ ') + name
            rows.append((label, tree[path]))
            walk(path, '' if not parent else prefix + ('   ' if last else '│  '))

    walk('', '')
    width = max(len(label) for label, _ in rows) + 2
    header = f"{'stage':<{width}}{'calls':>7}{'ms/call':>10}{'ms/plan':>10}{'% plan':>8}{'host ms/plan':>14}"
    lines = [header, '-' * len(header)]
    for label, v in rows:
        lines.append(f"{label:<{width}}{fmt(v['calls_per_plan'], 0):>7}{fmt(v['gpu_ms_per_call'], 2):>10}"
                     f"{fmt(v['gpu_ms_per_plan']):>10}{fmt(100 * v['gpu_ms_per_plan'] / total):>8}"
                     f"{fmt(v['host_ms_per_plan']):>14}".rstrip())
    lines += ['-' * len(header), f"{'plan total':<{width}}{'1':>7}{'':>10}{total:>10.1f}{'100.0':>8}"]
    return '\n'.join(lines)


def eval_share_table(runs):
    """Share of the whole eval wall time (world.evaluate), all replans including the warmup one."""
    keys = sorted(runs)
    lines = ['| Part | ' + ' | '.join(f'{a} h{h} s (%)' for a, h in keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    rows = defaultdict(dict)
    for k in keys:
        walls, parts = [], defaultdict(list)
        for t in runs[k]:
            run = t['run']
            walls.append(run['eval_wall_ms'])
            stages = run['stages']
            parts['history encode (outside plans)'].append(run['encode_ms'])
            for group, paths in GROUPS.items():
                parts[group].append(sum(stages.get(p, 0.0) for p in paths))
            parts['planning glue (other)'].append(run['propose_ms'] - sum(
                sum(stages.get(p, 0.0) for p in paths) for paths in GROUPS.values()))
            parts['planning total'].append(run['propose_ms'])
            parts['outside planning (env step, render, eval loop)'].append(
                run['eval_wall_ms'] - run['propose_ms'] - run['encode_ms'])
        wall = float(np.mean(walls))
        for part, values in parts.items():
            rows[part][k] = (float(np.mean(values)) / 1000, 100 * float(np.mean(values)) / wall)
        rows['**eval wall time**'][k] = (wall / 1000, 100.0)
    for part, cells in rows.items():
        lines.append(f'| {part} | ' + ' | '.join(f'{cells[k][0]:.2f} ({cells[k][1]:.1f}%)' for k in keys) + ' |')
    return '\n'.join(lines)


def section(root):
    runs = load(root)
    seeds = {f'{arm} h{h}': len(v) for (arm, h), v in sorted(runs.items())}
    level = next(iter(runs.values()))[0]['detail']
    text = [f'# Detail level `{level}`', '', f'Source: `{root}`; seeds per config: {seeds}.', '',
            '## Stage tree, per plan']
    for (arm, h), timings in sorted(runs.items()):
        text += ['', f'{arm} h{h}', '', '```text', tree_text(timings), '```']
    text += ['', '## Per plan, grouped', '', grouped_table(runs),
             '', '## Share of total eval time', '', eval_share_table(runs)]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--out')
    args = parser.parse_args()
    text = ['GPU ms from CUDA events. Stage tree and grouped table: per plan, mean over seeds, first replan of each '
            'run dropped (warmup). Eval-time table: whole `world.evaluate` wall time, all replans included. '
            '`(unattributed)` = parent minus its timed children (glue: history slide, goal inputs, Python).']
    for root in args.roots:
        text += [''] + section(root)
    text = '\n'.join(text) + '\n'
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
