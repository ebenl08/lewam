"""Record real and imagined latents of any LeWAM planner."""

from pathlib import Path

import numpy as np
import torch

from lewam.eval.planners import LeWAMPlanner


def predicted_step(replan_step: int, k: int, frameskip: int) -> int:
    """Env step whose frame imagined dynamics step k (0-based) predicts."""
    return replan_step + frameskip * (k + 1)


class LatentTraceMixin:
    """Per replan: real z_t and the returned plan's imagined latents; planning unchanged."""

    def init_trace(self) -> None:
        assert self.model.state_head == "mse", "re-imagination is exact only for a deterministic state head"
        assert self.rollout_steps > 1, "one-step plans have no imagined trajectory"
        self.trace_real, self.trace_imag, self.trace_goal = [], [], {}
        self._replan_envs, self._plan_depth = [], 0

    def _propose(self, info_dict, replan, history, history_pad):
        self._replan_envs = list(replan)
        for row, env in enumerate(replan):
            self.trace_real.append((env, int(self._step[env]), history[row, ..., -1, :].cpu()))
        return super()._propose(info_dict, replan, history, history_pad)

    def _traced_plan(self, method: str, history, history_pad, z_goal, steps_to_goal, call_args: tuple):
        self._plan_depth += 1
        try:
            plan = getattr(super(), method)(*call_args)
        finally:
            self._plan_depth -= 1
        if self._plan_depth == 0:
            self._record(history, history_pad, z_goal, steps_to_goal, plan)
        return plan

    def _best_of_k(self, history, history_pad, z_goal, h_norm=None, steps_to_goal=None):
        call_args = (history, history_pad, z_goal, h_norm, steps_to_goal)
        return self._traced_plan("_best_of_k", history, history_pad, z_goal, steps_to_goal, call_args)

    def _gradient_plan(self, history, history_pad, z_goal, steps_to_goal):
        call_args = (history, history_pad, z_goal, steps_to_goal)
        return self._traced_plan("_gradient_plan", history, history_pad, z_goal, steps_to_goal, call_args)

    def _cem_plan(self, history, history_pad, z_goal, steps_to_goal):
        call_args = (history, history_pad, z_goal, steps_to_goal)
        return self._traced_plan("_cem_plan", history, history_pad, z_goal, steps_to_goal, call_args)

    def _steer_rollout(self, history, history_pad, z_goal, steps_to_goal):
        call_args = (history, history_pad, z_goal, steps_to_goal)
        return self._traced_plan("_steer_rollout", history, history_pad, z_goal, steps_to_goal, call_args)

    def _beam_plan(self, history, history_pad, z_goal, steps_to_goal):
        call_args = (history, history_pad, z_goal, steps_to_goal)
        return self._traced_plan("_beam_plan", history, history_pad, z_goal, steps_to_goal, call_args)

    def _record(self, history, history_pad, z_goal, steps_to_goal, plan) -> None:
        latents = self._imagine_latents(history, history_pad, plan)
        for row, env in enumerate(self._replan_envs):
            steps_left = float("nan") if steps_to_goal is None else float(steps_to_goal[row])
            self.trace_goal.setdefault(env, z_goal[row].cpu())
            self.trace_imag.append((env, int(self._step[env]), steps_left, latents[row].cpu()))

    @torch.no_grad()
    def _imagine_latents(self, history, history_pad, plan):
        """(R, steps, ..., z) imagined latents of plans (R, steps * frameskip, action_dim)."""
        plan = plan.reshape(plan.shape[0], -1, self.frameskip, plan.shape[-1])
        latents = []
        for step in range(plan.shape[1]):
            pred_states = self.model.predict_states(history, history_pad, plan[:, step])
            next_latent = self._next_latent(pred_states)
            latents.append(next_latent.squeeze(-2))
            history, history_pad = self._slide_history(history, history_pad, next_latent)
        return torch.stack(latents, 1)

    def save_trace(self, path: str | Path) -> None:
        """trace.npz: real rows (env, step, z), imagined rows (env, step, steps_to_goal, k, z), goal z per env."""
        imag_rows = [(env, step, steps_left, k, latents[k])
                     for env, step, steps_left, latents in self.trace_imag for k in range(latents.shape[0])]
        goal_envs = sorted(self.trace_goal)
        np.savez(
            path,
            real_env=np.array([row[0] for row in self.trace_real], dtype=np.int64),
            real_step=np.array([row[1] for row in self.trace_real], dtype=np.int64),
            real_z=torch.stack([row[2] for row in self.trace_real]).float().numpy(),
            imag_env=np.array([row[0] for row in imag_rows], dtype=np.int64),
            imag_step=np.array([row[1] for row in imag_rows], dtype=np.int64),
            imag_steps_to_goal=np.array([row[2] for row in imag_rows], dtype=np.float64),
            imag_k=np.array([row[3] for row in imag_rows], dtype=np.int64),
            imag_z=torch.stack([row[4] for row in imag_rows]).float().numpy(),
            goal_env=np.array(goal_envs, dtype=np.int64),
            goal_z=torch.stack([self.trace_goal[env] for env in goal_envs]).float().numpy(),
            frameskip=np.int64(self.frameskip),
            rollout_steps=np.int64(self.rollout_steps),
        )


def traced(planner: LeWAMPlanner) -> LeWAMPlanner:
    """Swap planner's class to Traced<Class>; adds recording only."""
    cls = type(planner)
    planner.__class__ = type(f"Traced{cls.__name__}", (LatentTraceMixin, cls), {})
    planner.init_trace()
    return planner
