"""P2c capture-and-replay benchmark for the stable-worldmodel planners: LeWM CEM and LeFlow → CEM (swm env).

Runs a PG eval entry point (src/eval.py for CEM, src/leflow/eval.py for LeFlow → CEM) with the given Hydra
overrides. At the --capture-call-th solver call (all envs, warm start from the previous plan) it captures the
solver inputs, replays them at every batch size in --batches by tiling the envs (solver.batch_size = batch, so one
call covers all envs), and exits without finishing the eval. Same measurements as bench_lewam.py. PG is only imported.

Usage (from playground/stable-worldmodel, PYTHONPATH=src):
  python <LW>/scripts/planner_comparison/bench_pg.py --entry src/eval.py --out <run.json> --batches 1 4 16 \
      [--nsys] -- --config-name=pusht solver=cem ... eval.num_eval=16 solver.batch_size=16
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / 'analysis' / 'planner_comparison'))
from opcount import count_call, is_oom, measure_ceilings, time_calls  # noqa: E402

from leflow.latent_planner import LearnedLatentPathSolver  # noqa: E402
from stable_worldmodel.planning.solver.cem import CEMSolver  # noqa: E402


def tile(value, rows):
    """Per-env entries repeated to len(rows), as the PG policy slices them (tensors, arrays, lists)."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value[torch.as_tensor(rows)] if isinstance(value, torch.Tensor) else value[rows]
    if isinstance(value, list):
        return [value[i] for i in rows]
    return value


def replay(solver, captured, batch):
    """One solver call over `batch` envs, all in one chunk."""
    rows = np.arange(batch) % captured['n_envs']
    info = {k: tile(v, rows) for k, v in captured['info'].items()}
    kwargs = {k: tile(v, rows) for k, v in captured['kwargs'].items()}
    init = None if captured['init_action'] is None else tile(captured['init_action'], rows)

    def call():
        solver.batch_size = batch
        return solver.solve(info, init_action=init, **kwargs)
    return call


def bench(solver, captured, args):
    result = dict(solver=type(solver).__name__, captured_envs=captured['n_envs'], points=[])
    if not args.nsys:
        result['ceilings'] = measure_ceilings()
    for batch in args.batches:
        point = dict(batch=batch)
        try:
            call = replay(solver, captured, batch)
            point['ms_per_call'] = time_calls(call, label=f'measure B={batch}' if args.nsys else None)
            if not args.nsys:
                point.update(count_call(call))
        except RuntimeError as error:
            if not is_oom(error):
                raise
            point['oom'] = True
            torch.cuda.empty_cache()
        result['points'].append(point)
        print('[bench]', {k: v for k, v in point.items() if k != 'ms_per_call'},
              f"ms/call {np.mean(point.get('ms_per_call', [np.nan])):.1f}", flush=True)
        if point.get('oom'):
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True, help='PG eval script: src/eval.py or src/leflow/eval.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument('--capture-call', type=int, default=2, help='solver call to capture (2 = warm-started)')
    parser.add_argument('--nsys', action='store_true')
    args, overrides = parser.parse_known_args()
    overrides = [o for o in overrides if o != '--']
    calls = {'n': 0}
    originals = {cls: cls.__call__ for cls in (CEMSolver, LearnedLatentPathSolver)}

    def make_capture(cls):
        def capture(self, info_dict, init_action=None, **kwargs):
            calls['n'] += 1
            if calls['n'] < args.capture_call:
                return originals[cls](self, info_dict, init_action=init_action, **kwargs)
            captured = dict(info=dict(info_dict), kwargs=dict(kwargs),
                            init_action=None if init_action is None else init_action.clone(),
                            n_envs=len(next(v for v in info_dict.values() if torch.is_tensor(v))))
            result = bench(self, captured, args)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(dict(entry=args.entry, overrides=overrides, **result), indent=1))
            print(f'[bench] wrote {args.out}', flush=True)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()  # flush the CUPTI buffers: os._exit skips the flush at exit
            os._exit(0)
        return capture

    for cls in originals:
        cls.__call__ = make_capture(cls)
    sys.path.insert(0, str(Path(args.entry).resolve().parent))  # as when the entry runs as a script (Hydra targets)
    sys.argv = [args.entry, *overrides]
    runpy.run_path(args.entry, run_name='__main__')
    raise SystemExit(f'eval finished before solver call {args.capture_call}')


if __name__ == '__main__':
    main()
