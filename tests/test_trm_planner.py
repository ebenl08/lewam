"""TRM objective: planners with a stub head against the goal-MSE planners, head checkpoints, the latent cache."""

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from lewam.eval.beam_tree.beam_planner import LeWAMBeamPlanner
from lewam.eval.episodes import img_transform
from lewam.eval.planners import LeWAMPlanner
from lewam.trm.cache import build_latent_cache, load_cache, save_cache, select_episodes, verify_cache
from lewam.trm.checkpoint import load_head, save_head
from lewam.trm.head import PairwiseDistanceHead
from lewam.trm.planner import LeWAMBeamTRMPlanner, LeWAMTRMPlanner
from test_beam_planner import Z_DIM, make_planner, planning_inputs   # pytest puts tests/ on sys.path


class MSEHead(nn.Module):
    """A 'head' equal to the goal-MSE, so TRM planners must reproduce the goal-MSE planners."""

    def forward(self, z_i, z_j):
        return ((z_i - z_j) ** 2).mean(-1)


class AntiHead(nn.Module):
    """A head that prefers latents far from the goal, so its picks differ from goal-MSE."""

    def forward(self, z_i, z_j):
        return -((z_i - z_j) ** 2).mean(-1)


def beam_kwargs(width, children):
    return dict(beam_width=width, beam_children=children, beam_score="goal", beam_g_weight=1.0, beam_grad=False,
                beam_prune="topk", beam_tau=1.0, beam_elite=True, beam_soft_seed=7, _beam_soft_rng=None,
                beam_spread_path=None, _search_count=0)


def make_trm_beam(head, scope, width=4, children=4):
    return make_planner(LeWAMBeamTRMPlanner, width * children, trm_head=head, trm_scope=scope,
                        _trm_scale_logged=True, **beam_kwargs(width, children))


def test_mse_head_beam_is_the_goal_beam():
    history, history_pad, z_goal, steps_to_goal = planning_inputs(3, [4.0, 2.0, 3.0])
    expected = make_planner(LeWAMBeamPlanner, 16, **beam_kwargs(4, 4))._beam_plan(history, history_pad, z_goal, steps_to_goal)
    for scope in ("all", "prune"):
        plan = make_trm_beam(MSEHead(), scope)._beam_plan(history, history_pad, z_goal, steps_to_goal)
        assert torch.equal(plan, expected), scope


def test_mse_head_best_of_k_is_best_of_k():
    history, history_pad, z_goal, steps_to_goal = planning_inputs(3, [4.0, 2.0, 3.0])
    expected = make_planner(LeWAMPlanner, 32)._best_of_k(history, history_pad, z_goal, None, steps_to_goal)
    planner = make_planner(LeWAMTRMPlanner, 32, trm_head=MSEHead(), trm_scope="all", _trm_scale_logged=True)
    assert torch.equal(planner._best_of_k(history, history_pad, z_goal, None, steps_to_goal), expected)


def test_anti_head_changes_the_pick():
    history, history_pad, z_goal, steps_to_goal = planning_inputs(3, [4.0, 2.0, 3.0])
    expected = make_planner(LeWAMPlanner, 32)._best_of_k(history, history_pad, z_goal, None, steps_to_goal)
    planner = make_planner(LeWAMTRMPlanner, 32, trm_head=AntiHead(), trm_scope="all", _trm_scale_logged=True)
    assert not torch.equal(planner._best_of_k(history, history_pad, z_goal, None, steps_to_goal), expected)


def test_prune_scope_keeps_goal_mse_for_the_pick_and_prunes_by_the_head():
    history, history_pad, z_goal, steps_to_goal = planning_inputs(2, [4.0, 3.0])
    planner = make_trm_beam(AntiHead(), "prune")
    z = torch.randn(5, Z_DIM)
    goal = torch.randn(5, Z_DIM)
    assert torch.equal(planner._goal_cost(z, goal), LeWAMPlanner._goal_cost(planner, z, goal))

    tree, deadline_cost = planner._beam_search(history, history_pad, z_goal, steps_to_goal)
    level = tree.levels[0]
    goal_rep = z_goal.repeat_interleave(planner.num_proposals, 0)
    mse = LeWAMPlanner._goal_cost(planner, level.latent.flatten(0, 1), goal_rep).view(2, -1)
    assert torch.equal(planner._node_score(level, mse), -mse)       # pruning score is the head's

    goal_tree, _ = make_trm_beam(AntiHead(), "all")._beam_search(history, history_pad, z_goal, steps_to_goal)
    goal_mse_tree, _ = make_planner(LeWAMBeamPlanner, 16, **beam_kwargs(4, 4))._beam_search(
        history, history_pad, z_goal, steps_to_goal)
    assert torch.equal(tree.levels[1].parent, goal_tree.levels[1].parent)        # same head pruning as scope all
    assert not torch.equal(tree.levels[1].parent, goal_mse_tree.levels[1].parent)


def test_prune_scope_needs_beam():
    with pytest.raises(AssertionError):
        LeWAMTRMPlanner.__init__(object.__new__(LeWAMTRMPlanner), None, None, trm_head="x", trm_scope="prune")


def test_head_cost_is_differentiable_through_the_latent():
    head = PairwiseDistanceHead(dim=Z_DIM).requires_grad_(False)
    planner = make_planner(LeWAMTRMPlanner, 4, trm_head=head, trm_scope="all", _trm_scale_logged=True)
    z = torch.randn(6, Z_DIM, requires_grad=True)
    planner._goal_cost(z, torch.randn(6, Z_DIM)).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_multi_view_cost_averages_the_goal_views():
    head = PairwiseDistanceHead(dim=Z_DIM)
    planner = make_planner(LeWAMTRMPlanner, 4, trm_head=head, trm_scope="all", _trm_scale_logged=True,
                           goal_view_indices=[0, 2])
    z, goal = torch.randn(5, 3, Z_DIM), torch.randn(5, 2, Z_DIM)
    expected = 0.5 * (head(z[:, 0], goal[:, 0]) + head(z[:, 2], goal[:, 1]))
    assert torch.allclose(planner._goal_cost(z, goal), expected)


def test_head_checkpoint_round_trip(tmp_path):
    head = PairwiseDistanceHead(dim=Z_DIM, hidden_dim=16)
    config = dict(_target_="lewam.trm.head.PairwiseDistanceHead", dim=Z_DIM, hidden_dim=16, depth=2, dropout=0.0)
    path = save_head(head, tmp_path / "run", config, "weights_epoch_1.pt")
    loaded = load_head(path)
    z_i, z_j = torch.randn(3, Z_DIM), torch.randn(3, Z_DIM)
    assert torch.equal(loaded(z_i, z_j), head(z_i, z_j))


class StubEncoder(nn.Module):
    z_dim = 3

    def encode(self, pixels):
        return pixels.mean((2, 3))


@pytest.fixture
def tiny_h5(tmp_path):
    lengths = np.array([5, 3, 7, 4])
    rng = np.random.default_rng(0)
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as h5:
        h5["pixels"] = rng.integers(0, 256, size=(lengths.sum(), 8, 8, 3), dtype=np.uint8)
        h5["ep_len"] = lengths
        h5["ep_offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        h5["state"] = rng.normal(size=(lengths.sum(), 4)).astype(np.float32)
    return path


def test_cache_encodes_episodes_in_order_across_batches(tiny_h5, tmp_path):
    transform = img_transform(8)
    model = StubEncoder()
    with h5py.File(tiny_h5, "r") as h5:
        episodes = np.array([0, 2, 3])
        cache = build_latent_cache(h5, model, episodes, transform, device="cpu", batch_size=4)
        frames = np.concatenate([h5["pixels"][0:5], h5["pixels"][8:15], h5["pixels"][15:19]])
        expected = torch.stack([model.encode(transform(f)[None])[0] for f in frames]).numpy()
        assert np.allclose(cache.emb, expected, atol=1e-6)
        assert np.array_equal(cache.lengths, [5, 7, 4]) and np.array_equal(cache.offsets, [0, 5, 12])
        assert np.array_equal(cache.episode_idx, np.repeat(episodes, [5, 7, 4]))
        assert np.array_equal(cache.columns["xy"], h5["state"][:][np.r_[0:5, 8:15, 15:19], :2])
        verify_cache(cache, h5, model, transform, "cpu", n_rows=6, batch_size=4)

    cache.meta.update(dataset="tiny.h5", checkpoint="stub")
    loaded = load_cache(save_cache(cache, tmp_path / "c.npz"), expect_checkpoint="stub", expect_dataset="tiny.h5")
    assert np.array_equal(loaded.emb, cache.emb)
    with pytest.raises(ValueError):
        load_cache(tmp_path / "c.npz", expect_checkpoint="other")


def test_select_episodes_is_a_fixed_sorted_subset():
    a, b = select_episodes(100, 10, seed=3), select_episodes(100, 10, seed=3)
    assert np.array_equal(a, b) and np.array_equal(a, np.sort(a)) and len(np.unique(a)) == 10
    assert np.array_equal(select_episodes(5, None, 0), np.arange(5))
