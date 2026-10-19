"""Planner latency: CUDA-event GPU time per replan, per plan, per planner stage and for history encoding."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from lewam.eval.planners import LeWAMPlanner


class SingleEnvPlanning:
    """Mixin: plan one env per call, so one timed call is one plan (as solver.batch_size=1 in stable-worldmodel)."""

    def _propose(self, info_dict, replan, history, history_pad):
        plans = [self._plan_one(info_dict, [i], history[row:row + 1], history_pad[row:row + 1])
                 for row, i in enumerate(replan)]
        return torch.cat(plans, 0)

    def _plan_one(self, info_dict, replan, history, history_pad):
        return super()._propose(info_dict, replan, history, history_pad)


class SingleEnvLeWAMPlanner(SingleEnvPlanning, LeWAMPlanner):
    """LeWAMPlanner that plans each replanning env on its own."""


# timed stages per detail level: planner methods, model methods, model submodules, and the inline `_stage` hook
COARSE = {'policy': {'_goal_latents': 'goal_encode', '_imagine_rollout': 'proposals',
                     '_rollout_cost_differentiable': 'refine_forward'}}
PLANNER = {'policy': {**COARSE['policy'], '_rollout_noise': 'noise', '_window_cost': 'cost'},
           'model': {'predict_actions': 'policy_sample', 'predict_states': 'dynamics_step'}, 'hook': True}
MODEL = {**PLANNER, 'model': {**PLANNER['model'], 'forward_mot': 'forward_mot'}}
MOT = {**MODEL, 'model': {**MODEL['model'], '_build_attn_mask': 'attn_mask'},
       'modules': {'action_out': 'action_head', 'state_out': 'state_head'}, 'blocks': True}
LEVELS = {None: COARSE, 'planner': PLANNER, 'model': MODEL, 'mot': MOT}


class ReplanTimer:
    """GPU ms per replan of `_propose`, per `_plan_one` call (single-env mode), per planner stage and of the history
    encoding before it. One replan = one get_action call that plans. The first `skip_replans` replans are timed but
    dropped from the statistics (warmup). `detail` picks the stages: None (coarse), 'planner', 'model' or 'mot'.
    """

    def __init__(self, policy, skip_replans: int = 1, detail: str | None = None):
        assert detail in LEVELS, detail
        self.skip_replans, self.detail = skip_replans, detail
        self.replans = []          # per replan: dict(plans, propose, plan_calls, encode, calls) of CUDA events
        self._current = None
        self._in_propose = False
        self._stack = []           # open stage names, so nested calls are recorded under their path
        self._open_modules = []
        level = LEVELS[detail]
        policy._propose = self._wrap_propose(policy._propose)
        policy._enc = self._wrap_encode(policy._enc)
        if hasattr(policy, '_plan_one'):
            policy._plan_one = self._wrap_plan_one(policy._plan_one)
        for method, stage in level['policy'].items():
            if hasattr(policy, method):
                setattr(policy, method, self._wrap_stage(stage, getattr(policy, method)))
        model = getattr(policy, 'model', None)
        for method, stage in level.get('model', {}).items():
            setattr(model, method, self._wrap_stage(stage, getattr(model, method)))
        for attr, stage in level.get('modules', {}).items():
            self._hook_span(stage, getattr(model, attr), getattr(model, attr))
        if level.get('blocks'):
            self._hook_span('blocks', model.blocks[0], model.blocks[-1])
        if level.get('hook'):
            policy._stage = self.stage

    @staticmethod
    def _events():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def _open_replan(self):
        if self._current is None:
            self._current = dict(plans=0, propose=None, plan_calls=[], encode=[], calls=[])
        return self._current

    def _open(self, name):
        self._stack.append(name)
        torch.cuda.nvtx.range_push(name)  # same label on the nsys timeline
        start, end = self._events()
        return '/'.join(self._stack), start, end, time.perf_counter()

    def _close(self, record):
        path, start, end, host_start = record
        end.record()
        torch.cuda.nvtx.range_pop()
        self._stack.pop()
        self._open_replan()['calls'].append((path, start, end, 1000 * (time.perf_counter() - host_start)))

    @contextmanager
    def stage(self, name):
        record = self._open(name)
        try:
            yield
        finally:
            self._close(record)

    def _wrap_stage(self, stage, fn):
        def timed(*args, **kwargs):
            with self.stage(stage):
                return fn(*args, **kwargs)
        return timed

    def _hook_span(self, stage, first, last):
        """Time from the forward of module `first` to the end of the forward of module `last`."""
        first.register_forward_pre_hook(lambda module, args: self._open_modules.append(self._open(stage)))
        last.register_forward_hook(lambda module, args, output: self._close(self._open_modules.pop()))

    def _wrap_encode(self, fn):
        def timed(*args, **kwargs):
            if self._in_propose:  # goal encoding inside a plan is part of that plan
                return fn(*args, **kwargs)
            torch.cuda.nvtx.range_push('encode_history')
            start, end = self._events()
            out = fn(*args, **kwargs)
            end.record()
            torch.cuda.nvtx.range_pop()
            self._open_replan()['encode'].append((start, end))
            return out
        return timed

    def _wrap_propose(self, fn):
        def timed(info_dict, replan, *args, **kwargs):
            current = self._open_replan()
            current['plans'] = len(replan)
            self._in_propose = True
            torch.cuda.nvtx.range_push(f'replan ({len(replan)} envs)')
            start, end = self._events()
            out = fn(info_dict, replan, *args, **kwargs)
            end.record()
            torch.cuda.nvtx.range_pop()
            self._in_propose = False
            current['propose'] = (start, end)
            self.replans.append(current)
            self._current = None
            return out
        return timed

    def _wrap_plan_one(self, fn):
        def timed(*args, **kwargs):
            torch.cuda.nvtx.range_push('plan')
            start, end = self._events()
            out = fn(*args, **kwargs)
            end.record()
            torch.cuda.nvtx.range_pop()
            self._current['plan_calls'].append((start, end))
            return out
        return timed

    @staticmethod
    def _row(replan) -> dict:
        elapsed = lambda pair: pair[0].elapsed_time(pair[1])
        calls = [(path, elapsed((start, end)), host) for path, start, end, host in replan['calls']]
        return dict(plans=replan['plans'], propose_ms=elapsed(replan['propose']),
                    plan_ms=[elapsed(p) for p in replan['plan_calls']],
                    encode_ms=sum(elapsed(e) for e in replan['encode']),
                    stage_ms={path: sum(ms for p, ms, _ in calls if p == path)
                              for path in dict.fromkeys(p for p, _, _ in calls) if '/' not in path},
                    calls=calls)

    @staticmethod
    def _tree(rows) -> dict:
        """Per stage path over `rows`: calls, total GPU / host ms, GPU ms per call (mean, median)."""
        per_path = {}
        for row in rows:
            for path, gpu, host in row['calls']:
                entry = per_path.setdefault(path, dict(gpu=[], host=[]))
                entry['gpu'].append(gpu)
                entry['host'].append(host)
        plans = sum(r['plans'] for r in rows)
        return {path: dict(calls=len(v['gpu']), calls_per_plan=len(v['gpu']) / plans,
                           gpu_ms=float(np.sum(v['gpu'])), host_ms=float(np.sum(v['host'])),
                           gpu_ms_per_plan=float(np.sum(v['gpu'])) / plans,
                           host_ms_per_plan=float(np.sum(v['host'])) / plans,
                           gpu_ms_per_call=float(np.mean(v['gpu'])), gpu_ms_per_call_median=float(np.median(v['gpu'])),
                           host_ms_per_call=float(np.mean(v['host'])))
                for path, v in per_path.items()}

    def summary(self, eval_wall_s: float | None = None) -> dict:
        """Per kept replan: plans, propose ms, per-plan ms, encode ms, stage ms; medians, IQRs, plans/s; stage tree
        over the kept replans; and run totals over all replans (warmup included) next to the eval wall time."""
        torch.cuda.synchronize()
        all_rows = [self._row(r) for r in self.replans]
        rows, warmup = all_rows[self.skip_replans:], all_rows[:self.skip_replans]
        per_plan = [ms for r in rows for ms in r['plan_ms']]
        end_to_end = [ms + r['encode_ms'] / r['plans'] for r in rows for ms in r['plan_ms']]
        total_plans = sum(r['plans'] for r in rows)
        total_ms = sum(r['propose_ms'] + r['encode_ms'] for r in rows)
        stats = lambda xs: dict(n=len(xs), median=float(np.median(xs)), mean=float(np.mean(xs)),
                                q25=float(np.percentile(xs, 25)), q75=float(np.percentile(xs, 75))) if xs else None
        run = dict(eval_wall_ms=None if eval_wall_s is None else 1000 * eval_wall_s,
                   replans=len(all_rows), plans=sum(r['plans'] for r in all_rows),
                   propose_ms=sum(r['propose_ms'] for r in all_rows),
                   encode_ms=sum(r['encode_ms'] for r in all_rows),
                   stages={path: v['gpu_ms'] for path, v in self._tree(all_rows).items()} if all_rows else {})
        tree = self._tree(rows) if total_plans else {}
        for row in all_rows:
            del row['calls']  # per-call records are summarised in `tree` and `run`
        return dict(replans_total=len(self.replans), replans_kept=len(rows), plans_kept=total_plans,
                    detail=self.detail, plan_ms=stats(per_plan), end_to_end_ms=stats(end_to_end),
                    replan_propose_ms=stats([r['propose_ms'] for r in rows]),
                    replan_encode_ms=stats([r['encode_ms'] for r in rows]),
                    plans_per_s=1000 * total_plans / total_ms if total_ms else None,
                    tree=tree, run=run, warmup_replans=warmup, replans=rows)

    def write(self, path: Path, eval_wall_s: float | None = None) -> None:
        Path(path).write_text(json.dumps(self.summary(eval_wall_s), indent=1))
