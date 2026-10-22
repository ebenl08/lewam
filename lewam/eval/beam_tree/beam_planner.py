"""Beam-search planning with LeWAM"""

import json

import numpy as np
import torch

from lewam.eval.beam_tree.beam_tree import BeamTree, TreeLevel
from lewam.eval.planners import LeWAMPlanner
from lewam.models.lewam import LeWAM


class LeWAMBeamPlanner(LeWAMPlanner):
    """
    Level-synchronous beam search over flow-policy samples (plan_mode=beam). Each depth imagines
    beam_width * beam_children rows per env in one model pass, keeps the beam_width lowest-scoring nodes and gives
    each beam_children children. The plan of each env's lowest goal-MSE node at its goal deadline is returned.

    Pruning scores (beam_score):
    - `goal`: goal-MSE of the node latent
    - `astar`: f = beam_g_weight * latent path length + RMS distance to the goal

    Parent selection (beam_prune):
    - `topk`: the beam_width lowest scores, each copied beam_children times
    - `soft`: width parents drawn with replacement from softmax(-normalised score / beam_tau)

    With beam_grad, the beam's leaves replace the independent policy rollouts of the gradient planner (grad_* args).
    """

    def __init__(
        self, model: LeWAM, cfg: dict, *args, beam_width: int = 8, beam_children: int = 4,
        beam_score: str = "goal", beam_g_weight: float = 1.0, beam_grad: bool = False,
        beam_prune: str = "topk", beam_tau: float = 1.0, beam_elite: bool = True, beam_soft_seed: int = 0,
        beam_spread_path: str | None = None, **kwargs
    ):
        """
        Args:
            model (LeWAM): the trained model (requires dynamics)
            cfg (dict): LeWAM checkpoint config
            beam_width (int): nodes kept per env at each depth
            beam_children (int): policy samples per kept node; rows per env = beam_width * beam_children
            beam_score (str): "goal" or "astar"
            beam_g_weight (float): weight on the latent path length in the A* score
            beam_grad (bool): refine the beam's leaves with LeWAMPlanner._gradient_plan
            beam_prune (str): "topk" or "soft"
            beam_tau (float): soft only: temperature on the per-env normalised score
            beam_elite (bool): soft only: the lowest-scoring node always survives once
            beam_soft_seed (int): soft only: seed of the resampling generator, separate from the flow noise
            beam_spread_path (str | None): append per-depth spread statistics of the nodes to this jsonl file
            *args, **kwargs: LeWAMPlanner's arguments (plan_mode and num_proposals are set here)
        """
        assert beam_score in ("goal", "astar"), beam_score
        assert beam_prune in ("topk", "soft"), beam_prune
        assert beam_tau > 0, beam_tau
        assert 1 <= beam_width and 1 <= beam_children, (beam_width, beam_children)
        kwargs.update(plan_mode="best_of_k", num_proposals=beam_width * beam_children)
        super().__init__(model, cfg, *args, **kwargs)
        assert self.plan_cost_window == 1, "beam scores single steps"
        assert self.rollout_steps > 1, "beam needs more than one dynamics step"
        self.plan_mode, self.type = "beam", "lewam_plan_beam"
        self.beam_width = beam_width
        self.beam_children = beam_children
        self.beam_score = beam_score
        self.beam_g_weight = beam_g_weight
        self.beam_grad = beam_grad
        self.beam_prune = beam_prune
        self.beam_tau = beam_tau
        self.beam_elite = beam_elite
        self.beam_soft_seed = beam_soft_seed
        self._beam_soft_rng = None
        self.beam_spread_path = beam_spread_path
        self._search_count = 0
        print(f"[plan] beam: width {beam_width}, children {beam_children}, score {beam_score}"
              + (f", g weight {beam_g_weight}" if beam_score == "astar" else "")
              + (f", soft prune tau {beam_tau} elite {beam_elite} seed {beam_soft_seed}" if beam_prune == "soft"
                 else ", top-k prune")
              + (", gradient refinement of the leaves" if beam_grad else ""), flush=True)

    def _propose(self, info_dict: dict, replan: list[int], history: torch.Tensor, history_pad: torch.Tensor) -> torch.Tensor:
        """The actions to execute for each replanning env; copies LeWAMPlanner._propose's horizon block."""
        z_goal = self._goal_latents(info_dict, replan)
        steps_to_goal = None
        if self.model.goal_cond:
            if self.horizon_schedule == "constant":  # goal rollout_steps ahead at every replan, target slides
                steps_left = np.full(len(replan), float(self.rollout_steps))
            else:                                    # countdown to the dataset goal time, target pinned
                steps_left = np.maximum(self._steps_left[replan], 1.0)
            steps_to_goal = torch.tensor(steps_left, device=self.device, dtype=torch.float32)
        if self.beam_grad:
            plan = self._gradient_plan(history, history_pad, z_goal, steps_to_goal)
        else:
            plan = self._beam_plan(history, history_pad, z_goal, steps_to_goal)
        return plan[:, :self.exec_actions_per_plan]

    def _node_score(self, level: TreeLevel, goal_cost: torch.Tensor) -> torch.Tensor:
        """Pruning score, lower is better: goal-MSE, or A* f = g weight * path length + RMS goal distance."""
        if self.beam_score == "goal":
            return goal_cost
        return self.beam_g_weight * level.path_cost + goal_cost.sqrt()

    def _topk_parents(self, score: torch.Tensor) -> torch.Tensor:
        """The beam_width lowest scores per env, in row order, each repeated beam_children times."""
        kept = score.topk(self.beam_width, dim=1, largest=False).indices.sort(1).values
        return kept.repeat_interleave(self.beam_children, 1)

    def _soft_generator(self, device: torch.device | str) -> torch.Generator:
        """Generator of the soft resampling draws, seeded with beam_soft_seed; independent of the flow noise."""
        device = torch.device(device)
        if self._beam_soft_rng is None or self._beam_soft_rng.device != device:
            self._beam_soft_rng = torch.Generator(device=device)
            self._beam_soft_rng.manual_seed(self.beam_soft_seed)
        return self._beam_soft_rng

    def _soft_parents(self, score: torch.Tensor) -> torch.Tensor:
        """Parents drawn with replacement from softmax(-normalised score / beam_tau), in row order."""
        n_envs, width = score.shape
        score_range = score - score.min(1, keepdim=True).values
        normalised = score_range / score.std(1, keepdim=True).clamp_min(1e-8)
        probs = torch.softmax(-normalised / self.beam_tau, dim=1)
        parents = torch.multinomial(probs, width, replacement=True, generator=self._soft_generator(self.device))
        if self.beam_elite:
            parents[:, 0] = score.argmin(1)
        return parents.sort(1).values

    def _step_dist(self, z_next: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """RMS latent distance of one imagined step, averaged over the goal views."""
        squared = (z_next - z_prev) ** 2
        if squared.dim() == 2:
            return squared.mean(-1).sqrt()
        return squared[:, self.goal_view_indices].mean(-1).sqrt().mean(-1)

    @staticmethod
    def _distinct_per_env(index: torch.Tensor) -> torch.Tensor:
        """(n_envs,) number of distinct values in each row of index (n_envs, width)."""
        ordered = index.sort(1).values
        return (ordered[:, 1:] != ordered[:, :-1]).sum(1) + 1

    def _beam_plan(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None
    ) -> torch.Tensor:
        """(n_envs, n_dyn_steps * frameskip, action_dim) the actions of each env's lowest goal-MSE node at its deadline."""
        tree, deadline_cost = self._beam_search(history, history_pad, z_goal, steps_to_goal)
        return tree.plan(deadline_cost.argmin(1))

    def _imagine_rollout(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor,
        steps_to_goal: torch.Tensor | None, candidate_actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Beam leaves in place of the independent policy rollouts, so _gradient_plan refines the beam (beam_grad)."""
        assert candidate_actions is None, "beam_grad refines beam leaves only"
        roots = slice(None, None, self.num_proposals)    # inputs arrive repeated num_proposals times per env
        tree, deadline_cost = self._beam_search(history[roots], history_pad[roots], z_goal[roots],
                                                None if steps_to_goal is None else steps_to_goal[roots])
        leaves = tree.plan(torch.arange(tree.width, device=self.device).expand(tree.n_envs, -1))
        return leaves.reshape(history.shape[0], len(tree.levels), self.frameskip, -1), deadline_cost.flatten()

    def _beam_search(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None
    ) -> tuple[BeamTree, torch.Tensor]:
        """
        Depth loop: model step, tree.expand, score, tree.prune (identity once an env's deadline is reached).

        Returns:
            tree (BeamTree): the searched tree
            deadline_cost (torch.Tensor): (n_envs, width) goal-MSE at the env's deadline of each last-level node's
                ancestor at that depth
        """
        n_envs, width = history.shape[0], self.num_proposals
        num_rows = n_envs * width
        tree = BeamTree(history, history_pad, width)
        goal_rep = z_goal.repeat_interleave(width, 0)
        steps_rep = steps_to_goal.repeat_interleave(width, 0) if steps_to_goal is not None else None
        n_dyn_steps = self._dyn_steps_needed(steps_to_goal)
        deadline_idx = self._goal_step_idx(steps_to_goal)
        if deadline_idx is None:
            deadline_idx = torch.full((n_envs,), n_dyn_steps - 1, dtype=torch.long, device=self.device)
        noise = self._rollout_noise(n_envs, actions_given=False)
        identity = torch.arange(width, device=self.device).expand(n_envs, -1)
        deadline_cost = torch.zeros(n_envs, width, device=self.device)
        final_branches = torch.zeros(n_envs, dtype=torch.long, device=self.device)

        for step in range(n_dyn_steps):
            action_noise, state_noise = noise[step]
            goal_inputs = self._goal_inputs(goal_rep, steps_rep, step, num_rows)
            actions = self.model.predict_actions(tree.history, tree.history_pad, action_noise=action_noise, **goal_inputs)
            actions = actions[:, :self.frameskip]
            pred_states = self.model.predict_states(tree.history, tree.history_pad, actions, state_noise=state_noise)
            next_latent = self._next_latent(pred_states)
            step_cost = self._step_dist(next_latent.squeeze(-2), tree.parent_latent())
            
            tree.expand(actions, next_latent, step_cost)

            level = tree.levels[-1]
            goal_cost = self._goal_cost(level.latent.flatten(0, 1), goal_rep).view(n_envs, width)
            at_deadline = deadline_idx == step
            deadline_cost = torch.where(at_deadline[:, None], goal_cost, deadline_cost)
            final_branches = torch.where(at_deadline, self._distinct_per_env(tree.root_child()), final_branches)
            if self.beam_spread_path is not None:
                self._record_spread(tree, goal_cost, deadline_idx)
            if step == n_dyn_steps - 1:
                break

            score = self._node_score(level, goal_cost)
            parents = self._topk_parents(score) if self.beam_prune == "topk" else self._soft_parents(score)
            past_deadline = (deadline_idx <= step)[:, None]   # deadline reached: keep each row's chain
            tree.prune(torch.where(past_deadline, identity, parents))

        self._record_beam_stats(deadline_cost.min(1).values, final_branches)
        self._search_count += 1
        return tree, deadline_cost

    @staticmethod
    def _pairwise_rms(points: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(n_envs,) mean RMS distance over distinct node pairs (n_envs, width, dim), optionally only masked pairs."""
        points = points.float()
        dim = points.shape[-1]
        distance = torch.cdist(points, points) / dim ** 0.5
        width = points.shape[1]
        mask = ~torch.eye(width, dtype=torch.bool, device=points.device).expand(points.shape[0], -1, -1)
        if pair_mask is not None:
            mask = mask & pair_mask
        pair_count = mask.sum((1, 2))
        total = (distance * mask).sum((1, 2))
        return torch.where(pair_count > 0, total / pair_count.clamp_min(1), torch.full_like(total, float("nan")))

    def _record_spread(self, tree: BeamTree, goal_cost: torch.Tensor, deadline_idx: torch.Tensor) -> None:
        """Append the spread of the last level's nodes (one jsonl row per env) to beam_spread_path."""
        level = tree.levels[-1]
        depth = len(tree.levels)
        n_envs, width = goal_cost.shape
        identity = torch.arange(width, device=self.device).expand(n_envs, -1)
        same_parent = level.parent[:, :, None] == level.parent[:, None, :]
        prefixes = tree.plan(identity).flatten(2)                  # (n_envs, width, depth * frameskip * action_dim)
        stats = dict(
            action_spread=self._pairwise_rms(level.actions.flatten(2)),
            sibling_action_spread=self._pairwise_rms(level.actions.flatten(2), same_parent),
            prefix_spread=self._pairwise_rms(prefixes),
            latent_spread=self._pairwise_rms(level.latent.flatten(2)),
            distinct_parents=self._distinct_per_env(level.parent),
            distinct_roots=self._distinct_per_env(tree.root_child()),
            goal_min=goal_cost.min(1).values,
            goal_mean=goal_cost.mean(1),
        )
        stats = {name: values.tolist() for name, values in stats.items()}   # one CPU-GPU sync per stat
        deadline = (deadline_idx + 1).tolist()
        with open(self.beam_spread_path, "a") as file:
            for env in range(n_envs):
                row = dict(search=self._search_count, env=env, depth=depth, deadline=deadline[env],
                           frozen=depth > deadline[env], **{name: values[env] for name, values in stats.items()})
                file.write(json.dumps(row) + "\n")

    def _record_beam_stats(self, final_cost: torch.Tensor, final_branches: torch.Tensor) -> None:
        """Running mean best deadline cost and distinct surviving depth-1 branches; printed every 50 envs."""
        n_envs = final_cost.shape[0]
        stats = self.__dict__.setdefault("_beam_stats", dict(n=0, best_cost=0.0, branches=0.0))
        stats["n"] += n_envs
        stats["best_cost"] += float(final_cost.sum())   # float() calls: CPU-GPU syncs
        stats["branches"] += float(final_branches.sum())
        if stats["n"] % 50 < n_envs:
            print(f"[beam] score={self.beam_score} width={self.beam_width} children={self.beam_children}: "
                  f"mean best cost {stats['best_cost'] / stats['n']:.4f}, mean surviving depth-1 branches "
                  f"{stats['branches'] / stats['n']:.2f}", flush=True)
