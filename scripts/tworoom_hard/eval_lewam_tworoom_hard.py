"""LeWAM on TwoRoom hard: manifest (episodic) evaluation with the goal offset as a configured deadline.

Start/goal pairs come from scripts/tworoom_hard/export_manifests.py (the episodes of the stable-worldmodel CEM runs
for the same seed). There is no ground-truth goal time, so the planner's goal horizon is eval.h0 (deadline in model
steps; eval.goal_offset_steps records it in env steps). Episode budget = world.max_episode_steps = eval.eval_budget.

  python scripts/tworoom_hard/eval_lewam_tworoom_hard.py policy=lewam-tworoom seed=8902 eval.h0=5 \
      eval.goal_offset_steps=25 eval.rollout_steps=5 output_dir=data/tworoom_hard/native_d25/8902
"""

import csv
import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "scripts"))
from eval_lewam import register_env  # noqa: E402

from lewam.eval.build import build_policy  # noqa: E402
from lewam.eval.episodes import img_transform  # noqa: E402
from lewam.eval.loaders import load_lewam  # noqa: E402
from lewam.envs.tworoom_hard_env import TwoRoomGoalWrapper  # noqa: E402


def load_manifest(path: Path) -> tuple[list[dict], dict]:
    """Per-env reset options and the per-row manifest arrays."""
    manifest = dict(np.load(path))
    options = [{"state": a, "target_state": t} for a, t in zip(manifest["agent_xy"], manifest["target_xy"])]
    return options, manifest


def episode_successes(metrics: dict, seed: int, n_envs: int) -> np.ndarray:
    """Success per manifest row: World.evaluate reports episodes in completion order, env i has seed + i."""
    env_idx = np.asarray(metrics["seeds"]) - seed
    assert sorted(env_idx.tolist()) == list(range(n_envs)), f"unexpected episode seeds {metrics['seeds']}"
    success = np.zeros(n_envs, dtype=bool)
    success[env_idx] = np.asarray(metrics["episode_successes"], dtype=bool)
    return success


def episode_distances(world: swm.World) -> tuple[np.ndarray, np.ndarray]:
    """Last and smallest agent-target distance of each env's episode (envs are frozen once done in 'wait' mode)."""
    last, smallest = [], []
    for env in world.envs.envs:
        while not isinstance(env, TwoRoomGoalWrapper):
            env = env.env
        last.append(env.last_distance)
        smallest.append(env.min_distance)
    return np.asarray(last), np.asarray(smallest)


def write_outputs(out: Path, cfg: DictConfig, metrics: dict, manifest: dict, success: np.ndarray,
                  final_distance: np.ndarray, min_distance: np.ndarray, wall_s: float) -> None:
    """config.json, results.txt (stable-worldmodel format) and episodes.csv (one row per manifest pair)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=1))
    (out / "results.txt").write_text(
        f"seed: {cfg.seed}\n"
        f"Metrics: {{'success_rate': {metrics['success_rate']}, 'episode_successes': "
        f"array([{', '.join(str(bool(s)) for s in success)}])}}\n"
        f"Eval time: {wall_s:.1f}s\n"
    )
    with (out / "episodes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "same_room", "requires_door", "distance", "success", "final_distance",
                         "min_distance"])
        for i in range(len(success)):
            writer.writerow([int(manifest["row"][i]), bool(manifest["same_room"][i]),
                             bool(manifest["requires_door"][i]), float(manifest["distance"][i]), bool(success[i]),
                             round(float(final_distance[i]), 2), round(float(min_distance[i]), 2)])


@hydra.main(version_base="1.3", config_path="../../configs/eval", config_name="tworoom_hard")
def main(cfg: DictConfig):
    register_env(cfg.world.env_name)
    eval_cfg = cfg.eval
    options, manifest = load_manifest(Path(eval_cfg.manifest_dir) / f"seed{cfg.seed}_n{eval_cfg.num_eval}.npz")
    assert len(options) == eval_cfg.num_eval, f"{len(options)} manifest rows for num_eval={eval_cfg.num_eval}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg = load_lewam(cfg.policy, which=cfg.seq_which)
    model = model.to(device).eval().requires_grad_(False)
    transform = {"pixels": img_transform(eval_cfg.img_size), "goal": img_transform(eval_cfg.img_size)}
    policy = build_policy(eval_cfg, model, model_cfg, cfg.seed, transform, goal_offsets=None)
    print(f"[eval] {type(policy).__name__}, {eval_cfg.horizon_schedule} schedule, h0={eval_cfg.h0}, "
          f"deadline {eval_cfg.goal_offset_steps} env steps, budget {eval_cfg.eval_budget}", flush=True)

    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.set_policy(policy)
    start_time = time.time()
    metrics = world.evaluate(episodes=len(options), seed=cfg.seed, options=options, reset_mode="wait")
    wall_s = time.time() - start_time
    success = episode_successes(metrics, cfg.seed, len(options))
    final_distance, min_distance = episode_distances(world)
    print(f"==== RESULTS ====\nsuccess_rate {metrics['success_rate']}, same-room "
          f"{100 * success[manifest['same_room']].mean():.1f}, cross-wall {100 * success[~manifest['same_room']].mean():.1f}")
    out = Path(cfg.output_dir) if cfg.output_dir else LW / "data" / "tworoom_hard" / "scratch" / str(cfg.seed)
    write_outputs(out, cfg, metrics, manifest, success, final_distance, min_distance, wall_s)
    print("wrote", out)


if __name__ == "__main__":
    main()
