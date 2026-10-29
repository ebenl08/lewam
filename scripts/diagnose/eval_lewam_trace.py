"""LeWAM eval with latent tracing: real and imagined latents per replan, task-state snapshots per step.

Scored like the reference runs: eval_lewam.py's world metrics, or AnyGoal criterion 3 for the Cube variants.

  python scripts/diagnose/eval_lewam_trace.py --config-name=stackt policy=lewam-pusht eval.goal_anchored=true \
      eval.goal_offset_steps=50 eval.eval_budget=100 eval.rollout_steps=10 eval.plan_mode=grad \
      output_dir=data/beam_smoke/stackt/goal_anchored_trace/native_grad/8902
"""

import functools
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
sys.path.insert(0, str(LW / "scripts" / "anygoal"))
from eval_lewam import register_env, write_outputs  # noqa: E402
from eval_lewam_anygoal_cube import CUBE_KEYS, SUCCESS_KEY, anygoal_cube_success  # noqa: E402

from lewam import views  # noqa: E402
from lewam.diagnose.adapters import ADAPTERS, SnapshotLog  # noqa: E402
from lewam.diagnose.trace import traced  # noqa: E402
from lewam.eval.build import build_policy  # noqa: E402
from lewam.eval.episodes import (  # noqa: E402
    get_dataset, img_transform, sample_eval_episodes, sample_goal_anchored_episodes,
)
from lewam.eval.loaders import load_lewam  # noqa: E402


def save_snapshots(world: swm.World, path: Path) -> None:
    """snapshots.npz: one (steps, state_dim) array of adapter snapshots per env, keyed env_<i>."""
    snapshots = {}
    for i, env in enumerate(world.envs.envs):
        while not isinstance(env, SnapshotLog):
            env = env.env
        snapshots[f"env_{i}"] = np.stack(env.snapshots)
    np.savez(path, **snapshots)


@hydra.main(version_base="1.3", config_path="../../configs/eval", config_name="stackt")
def main(cfg: DictConfig):
    env_name = str(cfg.world.env_name)
    register_env(env_name)
    eval_cfg = cfg.eval
    assert cfg.output_dir, "output_dir is required"
    assert env_name in ADAPTERS, f"no adapter for {env_name}"
    assert eval_cfg.mode == "lewam_plan", "tracing needs a planner"
    assert not (eval_cfg.full_traj or eval_cfg.task_only or eval_cfg.get("timing") or eval_cfg.log_states), \
        "full_traj, task_only, timing and log_states are eval_lewam.py only"
    cube_key = SUCCESS_KEY.get(env_name)
    if cube_key:
        from anygoal.envs.cube_hop.env import add_cube_evaluation_metrics, prepare_cube_evaluation

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
    policy = traced(build_policy(eval_cfg, model, model_cfg, cfg.seed, transform, None))
    adapter = ADAPTERS[env_name]()
    print(f"[eval] policy={type(policy).__name__} adapter={type(adapter).__name__} "
          f"success={cube_key or 'world'}", flush=True)

    extra_views = views.extra_cameras(views.columns(model_cfg.get("views")))
    world = swm.World(
        **cfg.world, image_shape=(224, 224), extra_wrappers=[functools.partial(SnapshotLog, adapter=adapter)],
        **({"extra_views": extra_views} if extra_views else {})
    )
    world.set_policy(policy)
    callables = OmegaConf.to_container(eval_cfg.callables, resolve=True)
    if cube_key:
        callables = prepare_cube_evaluation(world, callables)

    results_path = Path(cfg.output_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=eval_cfg.goal_offset_steps,
        eval_budget=eval_cfg.eval_budget, episodes_idx=episodes, callables=callables,
        video=results_path if eval_cfg.save_video else None,
    )
    wall_s = time.time() - start_time
    if cube_key:
        metrics = anygoal_cube_success(add_cube_evaluation_metrics(world, metrics), cube_key)
        cube_metrics = {key: metrics[key] for key in CUBE_KEYS}
        (results_path / "cube_metrics.json").write_text(json.dumps(cube_metrics, indent=1, default=float))

    policy.save_trace(results_path / "trace.npz")
    save_snapshots(world, results_path / "snapshots.npz")
    write_outputs(results_path, cfg, metrics, episodes, starts, wall_s)
    print(f"==== {eval_cfg.mode} RESULTS ({cube_key or 'world'}) ====")
    print(metrics)
    print("wrote", results_path)


if __name__ == "__main__":
    main()
