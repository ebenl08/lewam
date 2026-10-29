"""Latent diagnosis: trace mixin on stub planners, alignment of imagined steps, env adapters."""

import numpy as np
import pytest
import torch

from lewam.diagnose.adapters import CubeAdapter, PushTAdapter
from lewam.diagnose.references import LatentBank, build_bank, sample_bank_episodes
from lewam.diagnose.trace import predicted_step, traced
from lewam.eval.planners import LeWAMPlanner
from test_beam_planner import (  # pytest puts tests/ on sys.path
    ACTION_DIM, FRAMESKIP, Z_DIM, StubModel, make_beam_planner, make_planner, planning_inputs,
)


class GradStubModel(StubModel):
    """StubModel with the state-noise draw the gradient planner makes (unused by the mse head)."""

    def sample_state_noise(self, batch_size, device, generator=None):
        return torch.randn(batch_size, 1, Z_DIM, device=device, generator=generator)


def make_grad_planner():
    planner = make_planner(LeWAMPlanner, 8, plan_mode="grad", grad_steps=3, grad_lr=0.05, grad_clip=10.0,
                           grad_tr=0.0, grad_action_clip=3.0, grad_all_k=True)
    planner.model = GradStubModel()
    return planner


def propose_inputs(n_envs, steps_left):
    history, history_pad, z_goal, _ = planning_inputs(n_envs, steps_left)
    return history, history_pad, z_goal, np.asarray(steps_left, dtype=np.float64)


def run_propose(planner, steps_left, env_steps):
    history, history_pad, z_goal, steps = propose_inputs(len(steps_left), steps_left)
    planner._steps_left = steps
    planner._step = np.asarray(env_steps)
    planner._goal_latents = lambda info_dict, replan: z_goal[replan]
    return planner._propose({}, list(range(len(steps_left))), history, history_pad), history, z_goal


def deadline_cost(planner, z_goal):
    """Cost of each recorded plan at its deadline step, from the trace."""
    rows = planner.trace_imag
    latents = torch.stack([row[3] for row in rows])
    steps_left = torch.tensor([row[2] for row in rows])
    return planner._window_cost(latents, z_goal, planner._goal_step_idx(steps_left))


@pytest.mark.parametrize("replan_step, k, expected", [(0, 9, 50), (5, 8, 50), (45, 0, 50), (50, 0, 55), (95, 0, 100)])
def test_predicted_step_matches_the_worked_example(replan_step, k, expected):
    assert predicted_step(replan_step, k, 5) == expected


def test_traced_best_of_k_plans_the_same_and_records_its_cheapest_cost():
    planner_kwargs = dict(plan_mode="best_of_k")
    expected, _, _ = run_propose(make_planner(LeWAMPlanner, 16, **planner_kwargs), [4.0, 2.0, 3.0], [0, 5, 10])
    planner = traced(make_planner(LeWAMPlanner, 16, **planner_kwargs))
    actions, history, z_goal = run_propose(planner, [4.0, 2.0, 3.0], [0, 5, 10])
    assert torch.equal(actions, expected)

    assert [(row[0], row[1]) for row in planner.trace_real] == [(0, 0), (1, 5), (2, 10)]
    assert torch.equal(torch.stack([row[2] for row in planner.trace_real]), history[:, -1])
    assert [row[3].shape for row in planner.trace_imag] == [(4, Z_DIM)] * 3

    # the recorded plan is the cheapest of the 16 proposals under the same flow draws
    fresh = make_planner(LeWAMPlanner, 16, **planner_kwargs)
    steps = torch.tensor([4.0, 2.0, 3.0])
    _, cost = fresh._imagine_rollout(history.repeat_interleave(16, 0), torch.zeros(48, 2, dtype=torch.bool),
                                     z_goal.repeat_interleave(16, 0), steps.repeat_interleave(16, 0))
    assert torch.allclose(deadline_cost(planner, z_goal), cost.view(3, 16).min(1).values)


def test_traced_gradient_plan_plans_the_same_and_records_its_best_cost():
    expected, _, _ = run_propose(make_grad_planner(), [4.0, 2.0], [0, 5])
    planner = traced(make_grad_planner())
    actions, _, z_goal = run_propose(planner, [4.0, 2.0], [0, 5])
    assert torch.equal(actions, expected)
    assert torch.allclose(deadline_cost(planner, z_goal).sum(), torch.tensor(planner._grad_stats["best_cost"]))


def test_traced_beam_plans_the_same_and_records_once_per_replan():
    expected, _, _ = run_propose(make_beam_planner(width=4, children=2), [4.0, 2.0], [0, 5])
    planner = traced(make_beam_planner(width=4, children=2))
    actions, _, _ = run_propose(planner, [4.0, 2.0], [0, 5])
    assert torch.equal(actions, expected)
    assert len(planner.trace_imag) == 2


def test_beam_grad_records_only_the_outer_call():
    planner = traced(make_beam_planner(width=4, children=2, beam_grad=True))
    planner.grad_steps, planner.grad_all_k = 0, True
    run_propose(planner, [4.0, 2.0], [0, 5])
    assert len(planner.trace_imag) == 2 and planner._plan_depth == 0


def test_save_trace_writes_one_row_per_imagined_step(tmp_path):
    planner = traced(make_planner(LeWAMPlanner, 8, plan_mode="best_of_k"))
    run_propose(planner, [4.0, 2.0], [0, 5])
    planner.save_trace(tmp_path / "trace.npz")
    trace = np.load(tmp_path / "trace.npz")
    assert trace["imag_z"].shape == (2 * 4, Z_DIM)
    assert trace["imag_k"].tolist() == [0, 1, 2, 3] * 2
    assert trace["real_step"].tolist() == [0, 5] and trace["goal_env"].tolist() == [0, 1]


def test_trace_needs_a_deterministic_state_head():
    planner = make_planner(LeWAMPlanner, 8, plan_mode="best_of_k")
    planner.model.state_head = "flow"
    with pytest.raises(AssertionError, match="deterministic"):
        traced(planner)


def test_pusht_adapter_errors_at_tolerance():
    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0])
    state = goal + np.array([20.0, 0.0, 0.0, 0.0, -np.pi / 9])
    assert np.allclose(PushTAdapter().errors(state, goal), [1.0, 1.0])
    wrapped = goal.copy()
    wrapped[4] = 2 * np.pi - 0.01                   # 0.01 rad from the goal angle, across the wrap
    assert PushTAdapter().errors(wrapped, goal)[1] < 0.1


def test_cube_adapter_errors_at_tolerance():
    goal = np.array([0.4, 0.0, 0.2, 0.5, 0.4, 0.1, 0.02])
    state = goal + np.array([0.04, 0.0, 0.0, 0.10, 0.0, 0.02, 0.0])
    assert np.allclose(CubeAdapter().errors(state, goal), [1.0, 1.0, 0.5])


def test_adapter_goal_reads_dataset_columns():
    dataset = {"state": np.arange(14.0).reshape(2, 7),
               "proprio_effector_pos": np.ones((2, 3)), "proprio_effector_yaw": np.full((2, 1), 0.5),
               "privileged_block_0_pos": np.zeros((2, 3))}
    assert PushTAdapter().goal(dataset, 1).tolist() == [7.0, 8.0, 9.0, 10.0, 11.0]
    assert CubeAdapter().goal(dataset, 0).tolist() == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]


def test_knn_matches_brute_force_and_excludes_the_episode():
    generator = torch.Generator().manual_seed(0)
    bank = LatentBank(latents=torch.randn(50, Z_DIM, generator=generator), episodes=torch.arange(50) // 10,
                      steps=torch.arange(50) % 10, dataset="toy", checkpoint="stub")
    z = torch.randn(3, Z_DIM, generator=generator)
    brute = ((z[:, None] - bank.latents[None]) ** 2).mean(-1)
    assert torch.allclose(bank.knn(z, k=5), brute.topk(5, dim=1, largest=False).values.mean(1), atol=1e-5)
    kept = bank.episodes != 2
    expected = brute[:, kept].topk(5, dim=1, largest=False).values.mean(1)
    assert torch.allclose(bank.knn(z, exclude_episode=2, k=5), expected, atol=1e-5)


def test_bank_halves_are_disjoint_and_cover_n_frames():
    ep_len = np.full(40, 10)
    half0 = sample_bank_episodes(ep_len, 100, seed=0, half=0)
    half1 = sample_bank_episodes(ep_len, 100, seed=0, half=1)
    assert not set(half0) & set(half1)
    assert len(half0) == len(half1) == 10
    with pytest.raises(AssertionError, match="frames available"):
        sample_bank_episodes(ep_len, 300, seed=0, half=0)


class StubEncoder(torch.nn.Module):
    """Mean pixel value per channel, as a 3-d latent."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode(self, pixels):
        return pixels.mean((2, 3))


def test_build_bank_trims_to_n_frames_and_saves(tmp_path):
    import h5py
    path = tmp_path / "toy.h5"
    with h5py.File(path, "w") as data:
        data["ep_len"] = np.array([4, 3, 5])
        data["ep_offset"] = np.array([0, 4, 7])
        data["pixels"] = np.random.default_rng(0).integers(0, 255, (12, 8, 8, 3), dtype=np.uint8)
    bank = build_bank(StubEncoder(), path, "stub", n_frames=6, seed=1, img_size=8, batch_size=2)
    assert len(bank.latents) == len(bank.episodes) == 6
    first = bank.episodes[0].item()
    assert bank.steps[bank.episodes == first].tolist() == list(range(int((bank.episodes == first).sum())))
    bank.save(tmp_path / "bank.pt")
    assert torch.equal(LatentBank.load(tmp_path / "bank.pt").latents, bank.latents)
