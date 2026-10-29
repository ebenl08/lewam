"""LeWAM on AnyGoal's Cube variants, scored with AnyGoal's Cube metrics (anygoal/planning.py).

  python scripts/anygoal/eval_lewam_anygoal_cube.py --config-name=cube_hop policy=lewam-cube eval.rollout_steps=5 \
      output_dir=data/beam_smoke/cube_hop/anygoal_c3/native_grad/8902
"""

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registers the HDF5 format
import torch
from omegaconf import DictConfig, OmegaConf

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "scripts"))
from eval_lewam import register_env, write_outputs  # noqa: E402

from lewam import views  # noqa: E402
from lewam.eval.build import build_policy  # noqa: E402
from lewam.eval.episodes import (  # noqa: E402
    get_dataset, img_transform, sample_eval_episodes, sample_goal_anchored_episodes,
)
from lewam.eval.loaders import load_lewam  # noqa: E402

# criterion AnyGoal selects per env (resources/plan/toy_predictors/*.yaml, output.success_criterion)
SUCCESS_KEY = {
    "anygoal/CubeHop-v0": "cube_hop_criterion_3",
    "anygoal/CubeDiagonal-v0": "cube_hop_criterion_3",
}
CUBE_KEYS = ("cube_hop_criterion_3", "original_cube_criterion", "cube_evaluation_tolerances", "cube_final_errors")


def anygoal_cube_success(metrics: dict, success_key: str) -> dict:
    """Copy AnyGoal's chosen Cube criterion into success_rate / episode_successes."""
    assert success_key in metrics, f"no {success_key} in metrics: not a Cube env?"
    metrics["success_rate"] = metrics[success_key]["success_rate"]
    metrics["episode_successes"] = np.asarray(metrics[success_key]["episode_successes"])
    return metrics


@hydra.main(version_base="1.3", config_path="../../configs/eval", config_name="cube_hop")
def main(cfg: DictConfig):
    register_env(cfg.world.env_name)  # puts AnyGoal on sys.path
    from anygoal.envs.cube_hop.env import add_cube_evaluation_metrics, prepare_cube_evaluation

    eval_cfg = cfg.eval
    success_key = SUCCESS_KEY[str(cfg.world.env_name)]
    assert cfg.output_dir, "output_dir is required"
    assert not (eval_cfg.full_traj or eval_cfg.task_only or eval_cfg.get("timing") or eval_cfg.log_states), \
        "full_traj, task_only, timing and log_states are eval_lewam.py only"

    dataset = get_dataset(eval_cfg.dataset_name, list(cfg.dataset.keys_to_cache), cfg.cache_dir)
    if eval_cfg.goal_anchored:
        episodes, starts, _ = sample_goal_anchored_episodes(
            dataset, eval_cfg.num_eval, cfg.seed, eval_cfg.goal_offset_steps
        )
    else:
        episodes, starts, _ = sample_eval_episodes(
            dataset, eval_cfg.num_eval, cfg.seed, eval_cfg.goal_offset_steps,
            random_start=eval_cfg.random_start, min_episode_len=eval_cfg.min_episode_len
        )
    cfg.world.max_episode_steps = 2 * eval_cfg.eval_budget

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg = load_lewam(cfg.policy, which=cfg.seq_which)
    model = model.to(device).eval().requires_grad_(False)
    transform = {"pixels": img_transform(eval_cfg.img_size), "goal": img_transform(eval_cfg.img_size)}
    policy = build_policy(eval_cfg, model, model_cfg, cfg.seed, transform, None)
    print(f"[eval] eval mode={eval_cfg.mode} policy={type(policy).__name__} success={success_key}")

    extra_views = views.extra_cameras(views.columns(model_cfg.get("views")))
    world = swm.World(**cfg.world, image_shape=(224, 224), **({"extra_views": extra_views} if extra_views else {}))
    world.set_policy(policy)
    callables = prepare_cube_evaluation(world, OmegaConf.to_container(eval_cfg.callables, resolve=True))

    results_path = Path(cfg.output_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=eval_cfg.goal_offset_steps,
        eval_budget=eval_cfg.eval_budget, episodes_idx=episodes, callables=callables,
        video=results_path if eval_cfg.save_video else None,
    )
    wall_s = time.time() - start_time
    metrics = anygoal_cube_success(add_cube_evaluation_metrics(world, metrics), success_key)

    cube_metrics = {key: metrics[key] for key in CUBE_KEYS}
    (results_path / "cube_metrics.json").write_text(json.dumps(cube_metrics, indent=1, default=float))
    grasp = [env.unwrapped.diagonal_grasp_summary() for env in world.envs.envs
             if hasattr(env.unwrapped, "diagonal_grasp_summary")]
    if grasp:  # CubeDiagonal: AnyGoal's opposite-edge contact audit, per env
        (results_path / "grasp_summary.json").write_text(json.dumps(grasp, indent=1, default=float))
    write_outputs(results_path, cfg, metrics, episodes, starts, wall_s)
    print(f"==== {eval_cfg.mode} RESULTS ({success_key}) ====")
    print(metrics)
    print("wrote", results_path)


if __name__ == "__main__":
    main()
