"""Record real and imagined latents of any LeWAM planner.

Follows the logic of PG's utils.rollout.LatentDriftLogger: the real latent of every frame the policy receives
(the terminal frame included), each imagined latent queued until its env step arrives, and latent_drift.csv
with the same columns.
"""

import csv
import math
from pathlib import Path

import numpy as np
import torch

from lewam.eval.planners import LeWAMPlanner

DRIFT_COLUMNS = (
    'replan', 't0', 'env', 'k', 't', 'executed', 'alive',
    'cost_pred', 'cost_real', 'drift_mse', 'prev_plan_mse',
)


def predicted_step(replan_step: int, k: int, frameskip: int) -> int:
    """Env step whose frame imagined dynamics step k (0-based) predicts."""
    return replan_step + frameskip * (k + 1)


class LatentTraceMixin:
    """Real z_t at every step, the returned plan's imagined latents per replan; planning unchanged."""

    def init_trace(self) -> None:
        assert self.model.state_head == "mse", "re-imagination is exact only for a deterministic state head"
        assert self.rollout_steps > 1, "one-step plans have no imagined trajectory"
        self.trace_real, self.trace_imag, self.trace_goal = [], [], {}
        self.drift_rows: list[dict] = []
        self._pending: dict[tuple[int, int], list[tuple[dict, torch.Tensor]]] = {}
        self._prev_plan: dict[int, tuple[int, torch.Tensor]] = {}   # env -> (t0, Z (steps + 1, ..., D))
        self._real_seen: set[tuple[int, int]] = set()
        self._replan_envs, self._plan_depth, self._replan_count = [], 0, 0

    # ---------------------------------------------------------------- step side
    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        self._record_real(info_dict)
        return super().get_action(info_dict, **kwargs)

    @torch.no_grad()
    def _record_real(self, info_dict: dict) -> None:
        """Encode each env's current frame once per (env, step), the frame it terminates on included."""
        if self._action_buffer is None:
            self.set_env(self.env)
        terminated = info_dict.get("terminated")
        dead = np.asarray(terminated, dtype=bool).reshape(-1) if terminated is not None \
            else np.zeros(self.env.num_envs, dtype=bool)
        envs = [env for env in range(self.env.num_envs) if (env, int(self._step[env])) not in self._real_seen]
        if not envs:
            return
        prepared = self._prepare_info(info_dict)
        frames = []
        for key in self.view_keys:
            current = prepared[key][envs]
            frames.append(current[:, -1] if current.ndim == 5 else current)
        pixels = torch.stack(frames, dim=1)                                     # (E, views, C, H, W)
        latents = self._enc(pixels.flatten(0, 1).to(self.device).float()).view(len(envs), len(self.view_keys), -1)
        if len(self.view_keys) == 1:
            latents = latents[:, 0]
        for row, env in enumerate(envs):
            step, alive = int(self._step[env]), not bool(dead[env])
            z_real = latents[row].cpu()
            self._real_seen.add((env, step))
            self.trace_real.append((env, step, z_real, alive))
            for drift_row, z_pred in self._pending.pop((env, step), []):
                self._fill_real(drift_row, env, z_pred, z_real, alive)
                self.drift_rows.append(drift_row)

    def _fill_real(self, row: dict, env: int, z_pred: torch.Tensor, z_real: torch.Tensor, alive: bool) -> None:
        """Fill cost_real / drift_mse / alive from the real latent of the row's step."""
        if env in self.trace_goal:
            row['cost_real'] = self._cost(z_real, self.trace_goal[env])
        row['drift_mse'] = float((z_pred.float() - z_real.float()).pow(2).sum())
        row['alive'] = alive

    # ---------------------------------------------------------------- plan side
    def _propose(self, info_dict, replan, history, history_pad):
        self._replan_envs = list(replan)
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

    def _cost(self, z: torch.Tensor, z_goal: torch.Tensor) -> float:
        """The planner's goal cost of one latent."""
        return float(self._goal_cost(z[None].to(self.device), z_goal[None].to(self.device))[0])

    def _record(self, history, history_pad, z_goal, steps_to_goal, plan) -> None:
        latents = self._imagine_latents(history, history_pad, plan)                 # (R, steps, ..., D)
        current = history[..., -1:, :] if history.dim() == 3 else history[..., -1:, :].transpose(1, 2)
        plan_latents = torch.cat([current.reshape(latents[:, :1].shape), latents], dim=1).cpu()  # index k -> ẑ_k
        executed = self.exec_actions_per_plan // self.frameskip
        for row, env in enumerate(self._replan_envs):
            t0 = int(self._step[env])
            steps_left = float("nan") if steps_to_goal is None else float(steps_to_goal[row])
            goal = z_goal[row].cpu()
            self.trace_goal.setdefault(env, goal)
            self.trace_imag.append((env, t0, steps_left, latents[row].cpu()))
            prev = self._prev_plan.get(env)
            for k in range(plan_latents.shape[1]):
                drift_row = {
                    'replan': self._replan_count, 't0': t0, 'env': env, 'k': k, 't': t0 + k * self.frameskip,
                    'executed': k <= executed, 'alive': math.nan, 'cost_pred': self._cost(plan_latents[row, k], goal),
                    'cost_real': math.nan, 'drift_mse': math.nan,
                    'prev_plan_mse': self._prev_plan_mse(prev, t0, k, plan_latents[row, k]),
                }
                if k == 0:  # sanity row: ẑ_0 is the encoded current frame
                    self._fill_real(drift_row, env, plan_latents[row, 0], plan_latents[row, 0], True)
                    self.drift_rows.append(drift_row)
                else:
                    self._pending.setdefault((env, drift_row['t']), []).append((drift_row, plan_latents[row, k]))
            self._prev_plan[env] = (t0, plan_latents[row])
        self._replan_count += 1

    def _prev_plan_mse(self, prev, t0: int, k: int, z: torch.Tensor) -> float:
        """Squared distance of ẑ_k to the previous plan's latent for the same env step."""
        if prev is None:
            return math.nan
        t0_prev, plan_prev = prev
        delta, rem = divmod(t0 - t0_prev, self.frameskip)
        if rem != 0 or k + delta >= plan_prev.shape[0]:
            return math.nan
        return float((z - plan_prev[k + delta]).pow(2).sum())

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

    # ---------------------------------------------------------------- output
    def save_trace(self, path: str | Path) -> None:
        """trace.npz (real rows per step, imagined rows per replan and k, goal z per env) + latent_drift.csv beside it."""
        path = Path(path)
        imag_rows = [(env, step, steps_left, k, latents[k])
                     for env, step, steps_left, latents in self.trace_imag for k in range(latents.shape[0])]
        goal_envs = sorted(self.trace_goal)
        np.savez(
            path,
            real_env=np.array([row[0] for row in self.trace_real], dtype=np.int64),
            real_step=np.array([row[1] for row in self.trace_real], dtype=np.int64),
            real_z=torch.stack([row[2] for row in self.trace_real]).float().numpy() if self.trace_real
            else np.zeros((0,), dtype=np.float32),
            real_alive=np.array([row[3] for row in self.trace_real], dtype=bool),
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
        rows = self.drift_rows + [row for items in self._pending.values() for row, _ in items]  # past the budget: NaN
        rows.sort(key=lambda row: (row['replan'], row['env'], row['k']))
        with (path.parent / "latent_drift.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(DRIFT_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)


def traced(planner: LeWAMPlanner) -> LeWAMPlanner:
    """Swap planner's class to Traced<Class>; adds recording only."""
    cls = type(planner)
    planner.__class__ = type(f"Traced{cls.__name__}", (LatentTraceMixin, cls), {})
    planner.init_trace()
    return planner
