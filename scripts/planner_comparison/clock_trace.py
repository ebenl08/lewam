"""P1-M3 clock trace: the planner's goal clock at every replan of one real eval run.

Wraps LeWAMPlanner._propose / _gradient_plan (no change to the eval code) and logs per replanning env: env step,
policy countdown steps_left, steps_to_goal handed to the planner, scored block, blocks imagined, h_norm at the first
imagined block, actions executed. Every other argument goes to scripts/eval_lewam.py unchanged.

CLI (from playground/lewam/):
    python scripts/planner_comparison/clock_trace.py --trace-out <file> --config-name=pusht policy=lewam-pusht ...
"""

import functools
import runpy
import sys
from pathlib import Path

from lewam.eval.planners import LeWAMPlanner

EVAL_SCRIPT = Path(__file__).resolve().parents[1] / 'eval_lewam.py'

ROWS = []


def traced_propose(propose):
    @functools.wraps(propose)
    def wrapper(self, info_dict, replan, history, history_pad):
        self._trace_replan = list(replan)
        self._trace_steps_left = [float(self._steps_left[i]) for i in replan]
        return propose(self, info_dict, replan, history, history_pad)
    return wrapper


def traced_gradient_plan(gradient_plan):
    @functools.wraps(gradient_plan)
    def wrapper(self, history, history_pad, z_goal, steps_to_goal):
        score = self._goal_step_idx(steps_to_goal) + 1
        imagined = self._dyn_steps_needed(steps_to_goal)
        for row, env in enumerate(self._trace_replan):
            steps = float(steps_to_goal[row])
            ROWS.append((env, int(self._step[env]), self._trace_steps_left[row], steps, int(score[row]), imagined,
                         min(max(steps, 1.0), self.H_max) / self.H_max, self.exec_actions_per_plan))
        return gradient_plan(self, history, history_pad, z_goal, steps_to_goal)
    return wrapper


def main():
    out = Path(sys.argv[sys.argv.index('--trace-out') + 1])
    del sys.argv[sys.argv.index('--trace-out'):sys.argv.index('--trace-out') + 2]
    LeWAMPlanner._propose = traced_propose(LeWAMPlanner._propose)
    LeWAMPlanner._gradient_plan = traced_gradient_plan(LeWAMPlanner._gradient_plan)
    runpy.run_path(str(EVAL_SCRIPT), run_name='__main__')  # as a script, so Hydra finds configs/eval next to it
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ['env t steps_left steps_to_goal score_block imagined h_norm exec']
    lines += [f'{env} {t} {left:g} {steps:g} {score} {imagined} {h_norm:.2f} {exe}'
              for env, t, left, steps, score, imagined, h_norm, exe in sorted(ROWS)]
    out.write_text('\n'.join(lines) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
