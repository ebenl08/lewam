"""Planner comparison additions: horizon_schedule, plan_cost_window, the results.txt writer and its loader."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lewam.eval.planners import LeWAMPlanner
from lewam.eval.timing import ReplanTimer, SingleEnvPlanning

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis" / "planner_comparison"))
from stats import episode_vectors  # noqa: E402


class _GoalModel:
    goal_cond = True


def bare_planner(rollout_steps=5, plan_cost_window=1, horizon_schedule="adaptive", H_max=10, frameskip=5):
    """A LeWAMPlanner with only the attributes the tested methods read (no checkpoint)."""
    planner = object.__new__(LeWAMPlanner)
    planner.model = _GoalModel()
    planner.device = torch.device("cpu")
    planner.rollout_steps, planner.plan_cost_window = rollout_steps, plan_cost_window
    planner.horizon_schedule, planner.H_max, planner.frameskip = horizon_schedule, H_max, frameskip
    planner.plan_goal_time, planner.plan_mode = True, "grad"
    planner.exec_actions_per_plan = frameskip
    return planner


def captured_steps_to_goal(planner, steps_left):
    """steps_to_goal that _propose hands to the gradient planner."""
    captured = {}
    planner._steps_left = np.asarray(steps_left, dtype=np.float64)
    planner._goal_latents = lambda info_dict, replan: torch.zeros(len(replan), 4)

    def fake_plan(history, history_pad, z_goal, steps_to_goal):
        captured["steps"] = steps_to_goal.clone()
        return torch.zeros(len(history), 50, 2)

    planner._gradient_plan = fake_plan
    replan = list(range(len(steps_left)))
    planner._propose({}, replan, torch.zeros(len(replan), 3, 4), torch.zeros(len(replan), 3, dtype=torch.bool))
    return captured["steps"]


@pytest.mark.parametrize("rollout_steps", [2, 5, 10])
def test_constant_schedule_scores_block_h_every_replan(rollout_steps):
    planner = bare_planner(rollout_steps=rollout_steps, horizon_schedule="constant")
    steps = captured_steps_to_goal(planner, [5.0, 3.0, 1.0, 1.0])
    assert torch.equal(steps, torch.full((4,), float(rollout_steps)))
    assert torch.equal(planner._goal_step_idx(steps), torch.full((4,), rollout_steps - 1))
    assert planner._dyn_steps_needed(steps) == rollout_steps
    inputs = [planner._goal_inputs(torch.zeros(4, 4), steps, step, 4)["h_norm"] for step in range(rollout_steps)]
    expected = [min(max(rollout_steps - step, 1), 10) / 10 for step in range(rollout_steps)]
    assert [float(h[0]) for h in inputs] == pytest.approx(expected)


def test_adaptive_schedule_is_the_countdown():
    planner = bare_planner(rollout_steps=5, horizon_schedule="adaptive")
    steps = captured_steps_to_goal(planner, [5.0, 3.0, 1.0, 0.5])
    assert torch.equal(steps, torch.tensor([5.0, 3.0, 1.0, 1.0]))
    assert torch.equal(planner._goal_step_idx(steps), torch.tensor([4, 2, 0, 0]))
    assert planner._dyn_steps_needed(steps) == 5


def test_window_one_is_the_scoring_step_cost():
    torch.manual_seed(0)
    latents, z_goal = torch.randn(6, 5, 8), torch.randn(6, 8)
    planner = bare_planner(plan_cost_window=1)
    for step_idx in (None, torch.tensor([0, 1, 2, 3, 4, 4])):
        expected = planner._goal_cost(planner._pred_state_at(latents, step_idx), z_goal)
        assert torch.equal(planner._window_cost(latents, z_goal, step_idx), expected)


def test_window_mean_over_valid_steps():
    torch.manual_seed(0)
    latents, z_goal = torch.randn(3, 5, 8), torch.randn(3, 8)
    planner = bare_planner(plan_cost_window=3)
    step_idx = torch.tensor([4, 1, 0])
    per_step = ((latents - z_goal[:, None]) ** 2).mean(-1)                  # (rows, steps)
    expected = torch.stack([per_step[0, 2:5].mean(), per_step[1, 0:2].mean(), per_step[2, 0]])
    assert torch.allclose(planner._window_cost(latents, z_goal, step_idx), expected)
    assert torch.allclose(planner._window_cost(latents, z_goal, None), per_step[:, 2:5].mean(1))


def test_results_round_trip(tmp_path):
    spec = importlib.util.spec_from_file_location("eval_lewam", REPO / "scripts" / "eval_lewam.py")
    eval_lewam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_lewam)
    successes = np.array([True, False, True, True] + [False] * 26)
    cfg = OmegaConf.create({"seed": 30531, "eval": {"num_eval": 30}})
    metrics = {"success_rate": 100 * successes.mean(), "episode_successes": successes}
    eval_lewam.write_outputs(tmp_path / "30531", cfg, metrics, list(range(30)), list(range(30)), 12.3)
    vectors = episode_vectors([str(tmp_path / "*")])
    assert list(vectors) == [30531]
    assert np.array_equal(vectors[30531], successes.astype(float))


class _FakePlanner:
    """Stands in for LeWAMPlanner: one row of actions per env, encodes the goal inside the plan."""

    def __init__(self):
        self.plan_calls = []

    def _enc(self, x):
        return x * 2

    def _propose(self, info_dict, replan, history, history_pad):
        self.plan_calls.append(list(replan))
        self._enc(history)  # goal encoding inside the plan
        return history[:, :1] + 1


class _FakeSingleEnvPlanner(SingleEnvPlanning, _FakePlanner):
    pass


def test_single_env_planning_one_call_per_env():
    planner = _FakeSingleEnvPlanner()
    history = torch.arange(12.0).view(4, 3)
    out = planner._propose({}, [0, 2, 5, 7], history, torch.zeros(4, 3, dtype=torch.bool))
    assert planner.plan_calls == [[0], [2], [5], [7]]
    assert torch.equal(out, history[:, :1] + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("single", [True, False])
def test_replan_timer_counts(single):
    planner = _FakeSingleEnvPlanner() if single else _FakePlanner()
    timer = ReplanTimer(planner, skip_replans=1)
    history = torch.ones(4, 3, device="cuda")
    for _ in range(3):  # get_action: history encoding, then the plan
        planner._enc(history)
        planner._propose({}, [0, 1, 2, 3], history, torch.zeros(4, 3, dtype=torch.bool, device="cuda"))
    summary = timer.summary()
    assert (summary["replans_total"], summary["replans_kept"], summary["plans_kept"]) == (3, 2, 8)
    assert all(len(row["plan_ms"]) == (4 if single else 0) for row in summary["replans"])
    assert summary["replan_encode_ms"]["n"] == 2       # goal encodings inside the plan are not counted
    assert (summary["plan_ms"] is not None) == single
    assert summary["plans_per_s"] > 0


@pytest.mark.skipif(not (Path.home() / ".stable_worldmodel" / "checkpoints" / "lewam-pusht").exists(),
                    reason="needs the lewam-pusht checkpoint")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rollout_steps,exec_actions", [(5, 5), (10, 10)])
def test_defaults_match_original_planner(rollout_steps, exec_actions):
    """adaptive + window 1 returns the same plans as the planner at git HEAD."""
    from lewam.eval.loaders import load_lewam

    source = subprocess.run(["git", "show", "HEAD:lewam/eval/planners.py"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout
    original = {}
    exec(compile(source, "planners_head.py", "exec"), original)

    model, cfg = load_lewam("lewam-pusht")
    model = model.cuda().eval().requires_grad_(False)
    kwargs = dict(frameskip=int(cfg["fs"]), action_dim=int(cfg["action_raw_dim"]), h0=np.array([5.0, 3.0, 1.0]),
                  flow_seed=0, plan_mode="grad", rollout_steps=rollout_steps, exec_actions_per_plan=exec_actions,
                  grad_all_k=True, grad_tr=0.01)
    plans = []
    for planner_class in (original["LeWAMPlanner"], LeWAMPlanner):
        planner = planner_class(model, cfg, **kwargs)
        planner.set_env(type("Env", (), {"num_envs": 3})())
        torch.manual_seed(0)
        history = torch.randn(3, int(cfg["history_len"]), model.z_dim, device="cuda")
        pad = torch.zeros(3, int(cfg["history_len"]), dtype=torch.bool, device="cuda")
        z_goal = torch.randn(3, model.z_dim, device="cuda")
        planner._goal_latents = lambda info_dict, replan, z_goal=z_goal: z_goal[replan]
        plans.append(planner._propose({}, [0, 1, 2], history, pad))
    assert torch.equal(plans[0], plans[1])
