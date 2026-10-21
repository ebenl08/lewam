"""Search tree over imagined latents, grown one level per model pass (used by LeWAMBeamPlanner)."""

from dataclasses import dataclass

import torch

from lewam.eval.planners import LeWAMPlanner


@dataclass
class TreeLevel:
    """One depth of the tree: `width` nodes per env."""
    parent: torch.Tensor      # (n_envs, width) node index in the previous level
    actions: torch.Tensor     # (n_envs, width, frameskip, action_dim) action block leading into the node
    latent: torch.Tensor      # (n_envs, width, [views,] z_dim) imagined latent of the node
    path_cost: torch.Tensor   # (n_envs, width) latent path length from the root


class BeamTree:
    """
    Search tree over imagined latents for n_envs roots. Every level holds `width` nodes per env; the frontier
    (history, history_pad) holds the model inputs of the rows imagined next, row = env * width + j.
    """

    def __init__(self, history: torch.Tensor, history_pad: torch.Tensor, width: int):
        """history, history_pad: (n_envs, ...) root inputs; each root starts with `width` children."""
        self.n_envs, self.width = history.shape[0], width
        self.history = history.repeat_interleave(width, 0)
        self.history_pad = history_pad.repeat_interleave(width, 0)
        self.levels: list[TreeLevel] = []
        device = history.device
        self._identity = torch.arange(width, device=device).expand(self.n_envs, -1)
        self._env_offset = torch.arange(self.n_envs, device=device)[:, None] * width
        self._root_latent = history.select(history.dim() - 2, -1)   # (n_envs, [views,] z_dim)
        # parent of each frontier row in the last level (the root before the first level)
        self._parents = torch.zeros(self.n_envs, width, dtype=torch.long, device=device)

    def _rows(self, index: torch.Tensor) -> torch.Tensor:
        """Flat frontier rows of per-env node indices (n_envs, width)."""
        return (self._env_offset + index).flatten()

    def expand(self, actions: torch.Tensor, next_latent: torch.Tensor, step_cost: torch.Tensor) -> None:
        """
        Add the frontier's imagined step as a new level and slide the frontier history.

        Args:
            actions (torch.Tensor): (n_envs * width, frameskip, action_dim) action block of each frontier row
            next_latent (torch.Tensor): (n_envs * width, [views,] 1, z_dim) imagined latent of each frontier row
            step_cost (torch.Tensor): (n_envs * width,) path cost of the step from the parent
        """
        n_envs, width = self.n_envs, self.width
        if self.levels:
            parent_cost = self.levels[-1].path_cost.gather(1, self._parents)
        else:
            parent_cost = torch.zeros(n_envs, width, device=step_cost.device)
        latent = next_latent.squeeze(-2)
        self.levels.append(TreeLevel(
            parent=self._parents,
            actions=actions.view(n_envs, width, *actions.shape[1:]),
            latent=latent.view(n_envs, width, *latent.shape[1:]),
            path_cost=parent_cost + step_cost.view(n_envs, width),
        ))
        self.history, self.history_pad = LeWAMPlanner._slide_history(self.history, self.history_pad, next_latent)
        self._parents = self._identity   # each frontier row continues its own node until pruned

    def prune(self, parents: torch.Tensor) -> None:
        """Overwrite frontier row j of env e with a copy of node parents[e, j] of the last level."""
        rows = self._rows(parents)
        self.history, self.history_pad = self.history[rows], self.history_pad[rows]
        self._parents = parents

    def parent_latent(self) -> torch.Tensor:
        """Latent of each frontier row's parent (the root latent before the first level)."""
        if not self.levels:
            return self._root_latent.repeat_interleave(self.width, 0)
        return self.levels[-1].latent.flatten(0, 1)[self._rows(self._parents)]

    def root_child(self) -> torch.Tensor:
        """(n_envs, width) depth-1 ancestor of each node in the last level."""
        node = self._identity
        for level in reversed(self.levels[1:]):
            node = level.parent.gather(1, node)
        return node

    def plan(self, node: torch.Tensor) -> torch.Tensor:
        """(n_envs, [k,] depth * frameskip, action_dim) actions from the root to node[e] (or node[e, i]) of the last level."""
        env_rows = torch.arange(self.n_envs, device=node.device)
        if node.dim() == 2:
            env_rows = env_rows[:, None]
        blocks = []
        for level in reversed(self.levels):
            blocks.append(level.actions[env_rows, node])
            node = level.parent[env_rows, node]
        return torch.stack(blocks[::-1], -3).flatten(-3, -2)
