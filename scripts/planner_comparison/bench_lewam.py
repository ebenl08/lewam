"""P2c capture-and-replay benchmark for the LeWAM planner (lewam env).

Runs scripts/eval_lewam.py with the given Hydra overrides. At the --capture-call-th replan (all envs replanning,
history full) it captures the planner inputs, replays them at every batch size in --batches by tiling the envs, and
exits without finishing the eval. Per batch: GPU ms per call (CUDA events), then, unless --nsys, one counted call
(FLOPs, op-level bytes, peak memory). With --nsys only the timed calls run, each inside an NVTX range
"measure B=<b> s=<steps>", for kernel time from nsys. Native (adaptive) runs once per --steps-left value.

Usage (from playground/lewam):
  python scripts/planner_comparison/bench_lewam.py --out <run.json> --batches 1 4 16 [--steps-left 5 4 3 2 1] \
      [--nsys] -- --config-name=pusht policy=lewam-pusht eval.plan_mode=grad ... eval.num_eval=16
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis' / 'planner_comparison'))
from opcount import ByteCounter, count_call, is_oom, measure_ceilings, time_calls  # noqa: E402

from lewam.eval.planners import LeWAMPlanner  # noqa: E402

PROPOSE = LeWAMPlanner._propose  # the unpatched method: replays must not re-enter the capture hook


def tile(value, rows):
    """Rows of a captured tensor / array, repeated to len(rows)."""
    return value[rows] if isinstance(value, (torch.Tensor, np.ndarray)) else value


def replay(planner, captured, batch, steps_left):
    """One plan call over `batch` envs: encode each env's newest frame, then _propose (goal encode + plan)."""
    n_envs = captured['history'].shape[0]
    rows = np.arange(batch) % n_envs
    info = {k: tile(v, rows) for k, v in captured['info'].items()}
    frames = captured['frames'][rows]
    history, pad = captured['history'][rows], captured['pad'][rows]
    replan = list(range(batch))

    def call():
        planner._steps_left = np.full(batch, float(steps_left))
        planner._enc(frames.flatten(0, 1).to(planner.device).float())   # steady state: one new frame per env
        return PROPOSE(planner, info, replan, history, pad)
    return call


def counters_preserve_plan(planner, call):
    """Same plan with and without the counting modes (flow RNG reseeded before each call)."""
    planner._flow_rng = None
    plain = call()
    planner._flow_rng = None
    with FlopCounterMode(display=False), ByteCounter():
        counted = call()
    return bool(torch.allclose(plain, counted, atol=1e-5))


def pass_check(planner, captured):
    """GFLOP of one differentiable WM rollout (5 blocks, 32 rows) forward, and forward + backward."""
    rows = np.zeros(planner.num_proposals, dtype=int)
    history, pad = captured['history'][rows], captured['pad'][rows]
    z_goal = planner._goal_latents(captured['info'], [0]).repeat_interleave(planner.num_proposals, 0)
    plan = torch.randn(planner.num_proposals, planner.rollout_steps, planner.frameskip,
                       planner.model.action_raw_dim, device=planner.device, requires_grad=True)
    steps = None

    def forward():
        with torch.enable_grad():
            return planner._rollout_cost_differentiable(history, pad, z_goal, steps, plan).sum()

    with FlopCounterMode(display=False) as fwd:
        forward()
    with FlopCounterMode(display=False) as both:
        forward().backward()
    return dict(rollout_forward_gflop=fwd.get_total_flops() / 1e9,
                rollout_backward_gflop=(both.get_total_flops() - fwd.get_total_flops()) / 1e9,
                wm_pass_gflop=fwd.get_total_flops() / 1e9 / planner.rollout_steps)


def bench(planner, captured, args):
    adaptive = planner.horizon_schedule != 'constant'
    steps_values = args.steps_left if adaptive else [planner.rollout_steps]
    result = dict(schedule=planner.horizon_schedule, captured_envs=int(captured['history'].shape[0]), points=[])
    if not args.nsys:
        result['ceilings'] = measure_ceilings()
        result['checks'] = dict(pass_check(planner, captured), counters_preserve_plan=counters_preserve_plan(
            planner, replay(planner, captured, 1, steps_values[0])))
        print('[bench] checks', result['checks'], flush=True)
    for steps in steps_values:
        for batch in args.batches:
            point = dict(batch=batch, steps_left=steps)
            try:
                call = replay(planner, captured, batch, steps)
                point['ms_per_call'] = time_calls(call, label=f'measure B={batch} s={steps}' if args.nsys else None)
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
                break  # larger batches will not fit either
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument('--steps-left', type=int, nargs='+', default=[5, 4, 3, 2, 1])
    parser.add_argument('--capture-call', type=int, default=2, help='replan call to capture (2 = history full)')
    parser.add_argument('--nsys', action='store_true')
    args, overrides = parser.parse_known_args()
    overrides = [o for o in overrides if o != '--']
    calls = {'n': 0}
    original = LeWAMPlanner._propose

    def capture(self, info_dict, replan, history, history_pad):
        calls['n'] += 1
        if calls['n'] < args.capture_call:
            return original(self, info_dict, replan, history, history_pad)
        frames = torch.stack([torch.as_tensor(info_dict[k][replan]) for k in self.view_keys], dim=1)
        frames = frames[:, :, -1] if frames.ndim == 6 else frames                  # newest frame per env
        captured = dict(info={k: info_dict[k][replan] for k in self.goal_keys}, frames=frames,
                        history=history.clone(), pad=history_pad.clone())
        result = bench(self, captured, args)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(overrides=overrides, **result), indent=1))
        print(f'[bench] wrote {args.out}', flush=True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()  # flush the CUPTI buffers: os._exit skips the flush at exit
        os._exit(0)  # the eval is not needed past the capture

    LeWAMPlanner._propose = capture
    sys.argv = [str(ROOT / 'scripts' / 'eval_lewam.py'), *overrides]
    runpy.run_path(sys.argv[0], run_name='__main__')
    raise SystemExit(f'eval finished before replan call {args.capture_call}')


if __name__ == '__main__':
    main()
