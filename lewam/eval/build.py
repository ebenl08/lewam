"""The LeWAM eval policy for `eval.mode`: lewam_policy runs it reactively, lewam_plan plans through its dynamics."""

import numpy as np
from omegaconf import DictConfig

from lewam.eval.beam_tree.beam_planner import LeWAMBeamPlanner
from lewam.eval.planners import LeWAMPlanner
from lewam.eval.policy import LeWAMPolicy
from lewam.eval.timing import SingleEnvLeWAMBeamPlanner, SingleEnvLeWAMPlanner
from lewam.models.lewam import LeWAM
from lewam.trm.planner import LeWAMBeamTRMPlanner, LeWAMTRMPlanner


def build_policy(
    eval_cfg: DictConfig, model: LeWAM, model_cfg: dict, seed: int, transform: dict,
    goal_offsets: list[int] | None = None
) -> LeWAMPolicy:
    """
    Args:
        eval_cfg (DictConfig): the `eval` section of the eval config
        model (LeWAM): the trained model, on its device
        model_cfg (dict): the checkpoint config
        seed (int): seed of the flow-matching draws
        transform (dict): per-key transforms applied to the info dict
        goal_offsets (list[int] | None): each episode's goal offset in env steps (full-trajectory protocol)

    Returns:
        policy (LeWAMPolicy): a LeWAMPolicy, or a LeWAMPlanner for lewam_plan
    """
    frameskip = int(model_cfg["fs"])
    h0 = eval_cfg.h0
    if h0 is None:
        h0 = (np.asarray(goal_offsets, np.float64) / frameskip if goal_offsets is not None
              else float(eval_cfg.goal_offset_steps) / frameskip)
    shared = dict(model=model, cfg=model_cfg, frameskip=frameskip, action_dim=int(model_cfg["action_raw_dim"]),
                  h0=h0, transform=transform, flow_seed=seed)

    if eval_cfg.mode == "lewam_policy":
        return LeWAMPolicy(num_exec_actions=eval_cfg.exec_actions, **shared)
    assert eval_cfg.mode == "lewam_plan", f"unknown eval.mode={eval_cfg.mode!r}: expected lewam_policy or lewam_plan"
    single_env = eval_cfg.get("timing") == "single"
    beam_args = {}
    if eval_cfg.plan_mode == "beam":
        planner_class = SingleEnvLeWAMBeamPlanner if single_env else LeWAMBeamPlanner
        beam_args = dict(beam_width=eval_cfg.beam_width, beam_children=eval_cfg.beam_children,
                         beam_score=eval_cfg.beam_score, beam_g_weight=eval_cfg.beam_g_weight,
                         beam_grad=eval_cfg.beam_grad, beam_prune=eval_cfg.beam_prune,
                         beam_tau=eval_cfg.beam_tau, beam_elite=eval_cfg.beam_elite,
                         beam_soft_seed=eval_cfg.beam_soft_seed, beam_spread_path=eval_cfg.beam_spread_path)
    else:
        planner_class = SingleEnvLeWAMPlanner if single_env else LeWAMPlanner
    trm_args = {}
    if eval_cfg.get("objective", "mse") == "trm":
        assert not single_env, "no single-env TRM planner"
        assert eval_cfg.trm_head, "eval.objective=trm needs eval.trm_head"
        planner_class = LeWAMBeamTRMPlanner if eval_cfg.plan_mode == "beam" else LeWAMTRMPlanner
        trm_args = dict(trm_head=eval_cfg.trm_head, trm_scope=eval_cfg.trm_scope)
    return planner_class(
        plan_mode=eval_cfg.plan_mode,
        num_proposals=eval_cfg.num_proposals,
        rollout_steps=eval_cfg.rollout_steps,
        exec_actions_per_plan=eval_cfg.exec_actions,
        plan_random_candidates=eval_cfg.plan_random_candidates,
        plan_goal_time=eval_cfg.plan_goal_time,
        horizon_schedule=eval_cfg.horizon_schedule,
        plan_cost_window=eval_cfg.plan_cost_window,
        grad_steps=eval_cfg.grad_steps,
        grad_lr=eval_cfg.grad_lr,
        grad_clip=eval_cfg.grad_clip,
        grad_tr=eval_cfg.grad_tr,
        grad_action_clip=eval_cfg.grad_action_clip,
        grad_all_k=eval_cfg.grad_all_k,
        steer_steps=eval_cfg.steer_steps,
        steer_lr=eval_cfg.steer_lr,
        steer_max_bias=eval_cfg.steer_max_bias,
        steer_k=eval_cfg.steer_k,
        steer_env_batch=eval_cfg.steer_env_batch,
        cem_iters=eval_cfg.cem_iters,
        cem_elites=eval_cfg.cem_elites,
        cem_std=eval_cfg.cem_std,
        cem_init=eval_cfg.cem_init,
        **beam_args,
        **trm_args,
        **shared)
