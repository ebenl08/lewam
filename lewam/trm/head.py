"""Trajectory Reachability Metrics: a learned pairwise distance over frozen latents.

`m_phi(z_i, z_j)` predicts the temporal separation `|t_i - t_j|` between two
states of the same logged trajectory -- a generic, oracle-free proxy for
reachability. It is trained on `(encode, encode)` pairs with the encoder and
dynamics frozen, and queried by the planner as `(predict, encode)`.

See li26, "Beyond Euclidean Proximity" (arXiv:2605.22164), and the design note
`notes/archive/trm-head-and-pair-sampling.md`. Ported verbatim from
`playground/stable-worldmodel/src/trm/head.py`.
"""

import torch
import torch.nn.functional as F
from torch import nn


class PairwiseDistanceHead(nn.Module):
    """m_phi(z_i, z_j) -> non-negative scalar predicting temporal separation.

    Feature map `psi = [z_i, z_j, z_i - z_j, |z_i - z_j|]` -- four D-dimensional
    blocks, hence the `4 * dim` input width -- then a `depth`-layer `hidden_dim`
    SiLU MLP with a Softplus scalar output (~262k parameters at the defaults,
    `dim=192`). The difference blocks hand the MLP the displacement instead of
    making it learn subtraction; the raw `z_i`, `z_j` let the metric depend on
    *where* in latent space the pair sits, which is what distinguishes a learned
    metric from a reweighted `||z_i - z_j||`.

    That feature map is deliberately ASYMMETRIC. The *label* is symmetric
    (`|t_i - t_j|` does not care about argument order), but three of the four
    blocks flip under a swap, so `m(a, b)` and `m(b, a)` are genuinely two
    different computations. Symmetry is taught by the sampler, which emits each
    pair in random order, rather than baked into the architecture -- an
    architecturally symmetric head (only `|z_i - z_j|`) would make
    `m(a, b) == m(b, a)` exactly but discard information the planner's one-sided
    query can use. That leaves the head only approximately symmetric, which is
    the right outcome: at plan time one argument is a predicted terminal latent
    and the other an encoded goal, so they are not interchangeable.
    """

    def __init__(self, dim=192, hidden_dim=256, depth=2, dropout=0.0):
        super().__init__()
        if depth < 1:
            raise ValueError(f'depth must be >= 1, got {depth}')
        # in_dim includes: z_i, z_j, z_i - z_j and |z_i - z_j|.
        layers, in_dim = [], 4 * dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.dim = dim

    def features(self, z_i, z_j):
        """(..., D) x (..., D) -> (..., 4D) feature map psi."""
        d = z_i - z_j
        return torch.cat([z_i, z_j, d, d.abs()], dim=-1)

    def forward(self, z_i, z_j):
        """(..., D) x (..., D) -> (...). Broadcasts, so (B, S, D) works directly.

        The two arguments are broadcast against each other, so a single goal
        latent can be scored against a whole candidate batch without an explicit
        `expand`.
        """
        if z_i.shape[-1] != self.dim or z_j.shape[-1] != self.dim:
            raise ValueError(
                f'expected trailing dim {self.dim}, got {tuple(z_i.shape)} and '
                f'{tuple(z_j.shape)}'
            )
        # broadcast the goal and candidate batch to have common B,S,D shape
        # The goal only has B, 1, D shape, so it needs broadcasting
        z_i, z_j = torch.broadcast_tensors(z_i, z_j)
        return F.softplus(self.net(self.features(z_i, z_j))).squeeze(-1)

    @torch.no_grad()
    def asymmetry(self, z_i, z_j):
        """|m(z_i, z_j) - m(z_j, z_i)|, elementwise.
        
        Diagnostic only. Large values on held-out pairs, relative to the mean
        prediction, mean the sampler's random-order augmentation never took --
        a training red flag rather than a property of the deployment query.
        """
        return (self(z_i, z_j) - self(z_j, z_i)).abs()


__all__ = ['PairwiseDistanceHead']