"""LeWAM planners with a TRM head as the goal cost (eval.objective=trm)"""

import torch

from lewam.eval.beam_tree.beam_planner import LeWAMBeamPlanner
from lewam.eval.beam_tree.beam_tree import TreeLevel
from lewam.eval.planners import LeWAMPlanner
from lewam.trm.checkpoint import load_head


class TRMCostMixin:
    """Replaces the goal-MSE of _goal_cost with a frozen TRM head m(z, z_goal)."""

    def _init_trm(self, trm_head: str, trm_scope: str) -> None:
        assert trm_scope in ("all", "prune"), trm_scope
        self.trm_head = load_head(trm_head).to(self.device).eval().requires_grad_(False)
        self.trm_scope = trm_scope
        self._trm_scale_logged = False
        print(f"[plan] TRM objective: head {trm_head}, scope {trm_scope}", flush=True)

    def _trm_cost(self, z_at_goal: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """(B,) head distance of the imagined latent to the goal, averaged over the goal views."""
        if z_at_goal.dim() == 2:
            return self.trm_head(z_at_goal, z_goal)
        z_goal = z_goal[:, None] if z_goal.dim() == 2 else z_goal
        return self.trm_head(z_at_goal[:, self.goal_view_indices], z_goal).mean(-1)

    def _log_cost_scale(self, trm_cost: torch.Tensor, z_at_goal: torch.Tensor, z_goal: torch.Tensor) -> None:
        """Once: median TRM cost next to median goal-MSE, the scale the gradient planner sees."""
        self._trm_scale_logged = True
        mse = super()._goal_cost(z_at_goal.detach(), z_goal.detach())
        print(f"[trm] first cost call: median TRM {trm_cost.detach().median().item():.4f}, "
              f"median goal-MSE {mse.median().item():.4f}", flush=True)

    def _goal_cost(self, z_at_goal: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        if self.trm_scope != "all":
            return super()._goal_cost(z_at_goal, z_goal)
        cost = self._trm_cost(z_at_goal, z_goal)
        if not self._trm_scale_logged:
            self._log_cost_scale(cost, z_at_goal, z_goal)
        return cost


class LeWAMTRMPlanner(TRMCostMixin, LeWAMPlanner):
    """LeWAMPlanner (grad, best_of_k, ...) scored by the TRM head."""

    def __init__(self, *args, trm_head: str, trm_scope: str = "all", **kwargs):
        assert trm_scope == "all", "trm_scope=prune needs plan_mode=beam"
        super().__init__(*args, **kwargs)
        self._init_trm(trm_head, trm_scope)


class LeWAMBeamTRMPlanner(TRMCostMixin, LeWAMBeamPlanner):
    """
    Beam planner with the TRM head:
    - `all`: pruning and the final pick at the deadline both by the head
    - `prune`: pruning by the head, final pick by deadline goal-MSE
    """

    def __init__(self, *args, trm_head: str, trm_scope: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self._init_trm(trm_head, trm_scope)

    def _beam_search(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None
    ):
        self._trm_goal_rep = z_goal.repeat_interleave(self.num_proposals, 0)   # goal of each node row
        return super()._beam_search(history, history_pad, z_goal, steps_to_goal)

    def _node_score(self, level: TreeLevel, goal_cost: torch.Tensor) -> torch.Tensor:
        if self.trm_scope == "all":
            return super()._node_score(level, goal_cost)                       # goal_cost is already the head
        return self._trm_cost(level.latent.flatten(0, 1), self._trm_goal_rep).view_as(goal_cost)
