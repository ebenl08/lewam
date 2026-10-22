"""Beam-search planner: BeamTree bookkeeping and LeWAMBeamPlanner against best_of_k on a stub model."""

import numpy as np
import pytest
import torch

import lewam.eval.beam_tree.beam_planner as beam_planner_module
from lewam.eval.beam_tree.beam_planner import LeWAMBeamPlanner
from lewam.eval.beam_tree.beam_tree import BeamTree
from lewam.eval.planners import LeWAMPlanner

Z_DIM, ACTION_DIM, CHUNK, FRAMESKIP = 4, 2, 10, 5


class StubModel:
    """Deterministic stand-in for LeWAM: actions from the history mean, goal and noise; linear dynamics."""
    goal_cond = True
    num_views = 1
    num_states_pred = 1
    state_head = "mse"
    z_dim = Z_DIM
    action_raw_dim = ACTION_DIM

    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.action_weight = torch.randn(Z_DIM, CHUNK * ACTION_DIM, generator=generator)
        self.state_weight = 0.3 * torch.randn(FRAMESKIP * ACTION_DIM, Z_DIM, generator=generator)

    def sample_action_noise(self, batch_size, device, generator):
        return torch.randn(batch_size, CHUNK, ACTION_DIM, device=device, generator=generator)

    def predict_actions(self, history, history_pad, action_noise=None, z_goal=None, h_norm=None):
        policy_mean = (history.mean(1) @ self.action_weight).view(-1, CHUNK, ACTION_DIM)
        goal_pull = z_goal[:, None, :ACTION_DIM] * h_norm[:, None, None]
        return policy_mean + goal_pull + action_noise

    def predict_states(self, history, history_pad, actions, state_noise=None):
        next_latent = history[:, -1] + actions.flatten(1) @ self.state_weight
        return next_latent[:, None]


def make_planner(planner_class, num_proposals, **attributes):
    """A planner with only the attributes the planning methods read (no checkpoint)."""
    planner = object.__new__(planner_class)
    planner.model = StubModel()
    planner.device = torch.device("cpu")
    planner.num_proposals = num_proposals
    planner.rollout_steps, planner.frameskip, planner.H_max = 4, FRAMESKIP, 10
    planner.exec_actions_per_plan = FRAMESKIP
    planner.plan_goal_time, planner.plan_cost_window = True, 1
    planner.plan_random_candidates = False
    planner.horizon_schedule = "adaptive"
    planner.goal_view_indices = [0]
    planner._flow_rng, planner._flow_seed = None, 7
    for name, value in attributes.items():
        setattr(planner, name, value)
    return planner


def make_beam_planner(width, children, score="goal", g_weight=1.0, beam_grad=False, prune="topk", tau=1.0, elite=True):
    return make_planner(LeWAMBeamPlanner, width * children, beam_width=width, beam_children=children,
                        beam_score=score, beam_g_weight=g_weight, beam_grad=beam_grad,
                        beam_prune=prune, beam_tau=tau, beam_elite=elite, beam_soft_seed=7, _beam_soft_rng=None,
                        beam_spread_path=None, _search_count=0)


def planning_inputs(n_envs, steps_to_goal):
    generator = torch.Generator().manual_seed(1)
    history = torch.randn(n_envs, 2, Z_DIM, generator=generator)
    history_pad = torch.zeros(n_envs, 2, dtype=torch.bool)
    z_goal = torch.randn(n_envs, Z_DIM, generator=generator)
    return history, history_pad, z_goal, torch.tensor(steps_to_goal)


@pytest.fixture
def captured_tree(monkeypatch):
    """The BeamTree built by the next _beam_plan call."""
    trees = []

    class RecordedTree(BeamTree):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trees.append(self)

    monkeypatch.setattr(beam_planner_module, "BeamTree", RecordedTree)
    return trees


# BeamTree alone

def test_tree_prune_copies_survivors_and_plan_backtracks():
    tree = BeamTree(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool), width=4)
    depth1_latent = torch.arange(4.0)[:, None, None].expand(4, 1, 3)    # node j at depth 1 has latent j
    depth1_actions = torch.arange(4.0)[:, None, None].expand(4, FRAMESKIP, ACTION_DIM)
    tree.expand(depth1_actions, depth1_latent, step_cost=torch.ones(4))

    tree.prune(torch.tensor([[1, 1, 3, 3]]))
    assert torch.equal(tree.history[:, -1, 0], torch.tensor([1.0, 1.0, 3.0, 3.0]))
    assert torch.equal(tree.parent_latent()[:, 0], torch.tensor([1.0, 1.0, 3.0, 3.0]))

    depth2_actions = (10 + torch.arange(4.0))[:, None, None].expand(4, FRAMESKIP, ACTION_DIM)
    tree.expand(depth2_actions, torch.zeros(4, 1, 3), step_cost=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(tree.levels[1].parent, torch.tensor([[1, 1, 3, 3]]))
    assert torch.equal(tree.levels[1].path_cost, torch.tensor([[2.0, 3.0, 4.0, 5.0]]))
    assert torch.equal(tree.root_child(), torch.tensor([[1, 1, 3, 3]]))

    plan = tree.plan(torch.tensor([2]))   # node 2 at depth 2: child of depth-1 node 3
    assert plan.shape == (1, 2 * FRAMESKIP, ACTION_DIM)
    assert torch.equal(plan[0, :FRAMESKIP, 0], torch.full((FRAMESKIP,), 3.0))
    assert torch.equal(plan[0, FRAMESKIP:, 0], torch.full((FRAMESKIP,), 12.0))


def test_tree_plan_of_several_nodes_per_env():
    tree = BeamTree(torch.zeros(2, 2, 3), torch.zeros(2, 2, dtype=torch.bool), width=3)
    for _ in range(2):
        tree.expand(torch.randn(6, FRAMESKIP, ACTION_DIM), torch.randn(6, 1, 3), torch.ones(6))
        tree.prune(torch.tensor([[2, 2, 0], [1, 0, 0]]))
    nodes = torch.tensor([[0, 2], [1, 1]])
    plans = tree.plan(nodes)
    assert plans.shape == (2, 2, 2 * FRAMESKIP, ACTION_DIM)
    for i in range(2):
        assert torch.equal(plans[:, i], tree.plan(nodes[:, i]))


def test_tree_without_prune_keeps_each_chain():
    tree = BeamTree(torch.zeros(2, 2, 3), torch.zeros(2, 2, dtype=torch.bool), width=3)
    for _ in range(3):
        tree.expand(torch.randn(6, FRAMESKIP, ACTION_DIM), torch.randn(6, 1, 3), torch.ones(6))
    assert torch.equal(tree.root_child(), torch.arange(3).expand(2, -1))
    assert torch.equal(tree.levels[-1].path_cost, torch.full((2, 3), 3.0))


# LeWAMBeamPlanner

def test_topk_parents_keeps_lowest_scores_in_row_order():
    planner = make_beam_planner(width=2, children=3)
    score = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.7, 0.3]])
    assert torch.equal(planner._topk_parents(score), torch.tensor([[1, 1, 1, 3, 3, 3]]))


def test_soft_parents_cold_temperature_copies_the_best():
    planner = make_beam_planner(width=2, children=3, prune="soft", tau=1e-3)
    score = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.7, 0.3]])
    assert torch.equal(planner._soft_parents(score), torch.full((1, 6), 1))


def test_soft_parents_keep_the_elite_and_row_order():
    planner = make_beam_planner(width=4, children=4, prune="soft", tau=100.0)
    score = torch.rand(3, 16, generator=torch.Generator().manual_seed(2))
    parents = planner._soft_parents(score)
    assert parents.shape == (3, 16)
    assert torch.equal(parents, parents.sort(1).values)
    for env in range(3):
        assert int(score[env].argmin()) in parents[env].tolist()
    assert len(set(parents[0].tolist())) > 4      # hot temperature: far more distinct parents than top-k's 4


def test_soft_resampling_leaves_the_flow_noise_untouched():
    planner = make_beam_planner(width=4, children=4, prune="soft")
    flow_state = planner._flow_generator(planner.device).get_state()
    planner._soft_parents(torch.rand(2, 16))
    assert torch.equal(planner._flow_generator(planner.device).get_state(), flow_state)


def test_pairwise_rms_is_the_mean_distance_over_distinct_pairs():
    points = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]])
    expected = (2 * 5 + 2 * 5 + 2 * 0) / 6 / 2 ** 0.5         # pairs (0,1),(1,0),(1,2),(2,1),(0,2),(2,0)
    assert torch.allclose(LeWAMBeamPlanner._pairwise_rms(points), torch.tensor([expected]))
    only_first_pair = torch.zeros(1, 3, 3, dtype=torch.bool)
    only_first_pair[0, 0, 1] = only_first_pair[0, 1, 0] = True
    assert torch.allclose(LeWAMBeamPlanner._pairwise_rms(points, only_first_pair), torch.tensor([5 / 2 ** 0.5]))


def test_spread_log_rows(tmp_path):
    import json
    path = tmp_path / "spread.jsonl"
    planner = make_beam_planner(width=32, children=1)
    planner.beam_spread_path = str(path)
    history, history_pad, z_goal, steps_to_goal = planning_inputs(2, [3.0, 2.0])
    planner._beam_plan(history, history_pad, z_goal, steps_to_goal)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 * 3                               # 2 envs x 3 depths
    depth2 = [row for row in rows if row["depth"] == 2]
    assert all(row["distinct_parents"] == 32 for row in depth2)          # independent chains
    assert all(row["sibling_action_spread"] is None or row["sibling_action_spread"] != row["sibling_action_spread"]
               for row in depth2)                                         # no siblings: nan
    assert [row["frozen"] for row in rows if row["env"] == 1] == [False, False, True]


def test_soft_prune_plans_end_to_end():
    history, history_pad, z_goal, steps_to_goal = planning_inputs(2, [4.0, 2.0])
    plan = make_beam_planner(width=4, children=4, prune="soft")._beam_plan(history, history_pad, z_goal, steps_to_goal)
    assert plan.shape == (2, 4 * FRAMESKIP, ACTION_DIM)


@pytest.mark.parametrize("score", ["goal", "astar"])
def test_one_child_per_node_is_best_of_k(score):
    history, history_pad, z_goal, steps_to_goal = planning_inputs(3, [4.0, 2.0, 3.0])
    expected = make_planner(LeWAMPlanner, 32)._best_of_k(history, history_pad, z_goal, None, steps_to_goal)
    plan = make_beam_planner(width=32, children=1, score=score)._beam_plan(history, history_pad, z_goal, steps_to_goal)
    assert torch.equal(plan, expected)


def test_beam_grad_without_steps_is_the_beam_plan():
    history, history_pad, z_goal, steps_to_goal = planning_inputs(3, [4.0, 2.0, 3.0])
    expected = make_beam_planner(width=4, children=2)._beam_plan(history, history_pad, z_goal, steps_to_goal)
    planner = make_beam_planner(width=4, children=2, beam_grad=True)
    planner.grad_steps, planner.grad_all_k = 0, True
    plan = planner._gradient_plan(history, history_pad, z_goal, steps_to_goal)
    assert torch.equal(plan, expected)


def test_pruning_follows_scores_and_freezes_after_deadline(captured_tree):
    planner = make_beam_planner(width=2, children=3)
    history, history_pad, z_goal, steps_to_goal = planning_inputs(2, [1.0, 3.0])
    plan = planner._beam_plan(history, history_pad, z_goal, steps_to_goal)
    tree = captured_tree[0]
    width = planner.num_proposals

    depth1_cost = planner._goal_cost(tree.levels[0].latent.flatten(0, 1), z_goal.repeat_interleave(width, 0)).view(2, width)
    # env 0 (deadline 1): its cheapest depth-1 node, and no pruning afterwards
    best0 = int(depth1_cost[0].argmin())
    assert torch.equal(plan[0, :FRAMESKIP], tree.levels[0].actions[0, best0])
    for level in tree.levels[1:]:
        assert torch.equal(level.parent[0], torch.arange(width))
    # env 1 (deadline 3): depth 2 descends from its two cheapest depth-1 nodes, three children each
    kept = depth1_cost[1].topk(2, largest=False).indices.sort().values
    assert torch.equal(tree.levels[1].parent[1], kept.repeat_interleave(3))


def test_astar_path_cost_is_rms_path_length(captured_tree):
    planner = make_beam_planner(width=2, children=2, score="astar")
    history, history_pad, z_goal, steps_to_goal = planning_inputs(1, [3.0])
    planner._beam_plan(history, history_pad, z_goal, steps_to_goal)
    tree = captured_tree[0]

    node = 0
    path_latents = []
    for level in reversed(tree.levels):
        path_latents.append(level.latent[0, node])
        node = int(level.parent[0, node])
    path_latents.append(history[0, -1])
    path = torch.stack(path_latents[::-1])
    rms_steps = ((path[1:] - path[:-1]) ** 2).mean(-1).sqrt()
    assert torch.allclose(tree.levels[-1].path_cost[0, 0], rms_steps.sum())


@pytest.mark.parametrize("schedule", ["adaptive", "constant"])
def test_propose_hands_the_base_steps_to_goal(schedule):
    steps_left = np.array([5.0, 3.0, 1.0, 0.5])
    replan = list(range(len(steps_left)))
    history, history_pad = torch.zeros(4, 2, Z_DIM), torch.zeros(4, 2, dtype=torch.bool)
    captured = {}

    def capture(name):
        def fake_plan(history, history_pad, z_goal, steps_to_goal, *args):
            captured[name] = steps_to_goal.clone()
            return torch.zeros(len(history), 50, ACTION_DIM)
        return fake_plan

    for name, planner in (("base", make_planner(LeWAMPlanner, 32, plan_mode="grad")), ("beam", make_beam_planner(8, 4))):
        planner.horizon_schedule = schedule
        planner._steps_left = steps_left
        planner._goal_latents = lambda info_dict, replan: torch.zeros(len(replan), Z_DIM)
        planner._gradient_plan = planner._beam_plan = capture(name)
        planner._propose({}, replan, history, history_pad)
    assert torch.equal(captured["beam"], captured["base"])
