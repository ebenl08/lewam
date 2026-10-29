"""LeWAM on PointMaze Large through the env harness: random start/goal pairs instead of dataset episodes.

Each seed draws eval.num_eval ordered pairs of free maze cells whose shortest route is at least eval.min_route_cells
cells, adds OGBench's reset noise to both positions, and runs every pair once (lewam/PointMazeHarness-v0).

  python scripts/paper_repro/eval_lewam_pointmaze_harness.py seed=42 eval.h0=10 eval.goal_offset_steps=50 \
      output_dir=data/paper_repro/pointmaze_harness/reactive_d50/42
"""

import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "scripts"))
sys.path.insert(0, str(LW / "scripts" / "tworoom_hard"))
from eval_lewam_tworoom_hard import episode_successes  # noqa: E402

from lewam.envs.pointmaze_harness_env import PointMazeHarnessEnv  # noqa: E402
from lewam.eval.build import build_policy  # noqa: E402
from lewam.eval.episodes import img_transform  # noqa: E402
from lewam.eval.loaders import load_lewam  # noqa: E402


def route_lengths(maze_map: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Shortest route in cells from start to every reachable free cell (BFS, 4-connected)."""
    length = {start: 0}
    queue = deque([start])
    while queue:
        i, j = queue.popleft()
        for neighbour in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if maze_map[neighbour] == 0 and neighbour not in length:
                length[neighbour] = length[(i, j)] + 1
                queue.append(neighbour)
    return length


def draw_pairs(env: PointMazeHarnessEnv, num_pairs: int, min_route_cells: int, seed: int) -> list[dict]:
    """num_pairs distinct ordered (start, goal) cell pairs with a route of at least min_route_cells, noised."""
    maze, maze_env = np.asarray(env.u.maze_map), env.u
    free = [tuple(c) for c in np.argwhere(maze == 0)]
    pairs = [(s, g, n) for s in free for g, n in route_lengths(maze, s).items() if n >= min_route_cells]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(pairs), size=num_pairs, replace=False)
    noise = maze_env._noise * maze_env._maze_unit / 4      # OGBench's add_noise: uniform +-1 unit
    rows = []
    for k in picks:
        start, goal, cells = pairs[k]
        start_xy = np.asarray(maze_env.ij_to_xy(start)) + rng.uniform(-noise, noise, 2)
        goal_xy = np.asarray(maze_env.ij_to_xy(goal)) + rng.uniform(-noise, noise, 2)
        rows.append({"start_xy": start_xy, "goal_xy": goal_xy, "route_cells": cells})
    return rows


def episode_distances(world: swm.World) -> tuple[np.ndarray, np.ndarray]:
    """Last and smallest goal distance of each env's episode."""
    last, smallest = [], []
    for env in world.envs.envs:
        while not isinstance(env, PointMazeHarnessEnv):
            env = env.env
        last.append(env.last_distance)
        smallest.append(env.min_distance)
    return np.asarray(last), np.asarray(smallest)


def rename_videos(out: Path, metrics: dict, seed: int) -> None:
    """World names videos by finishing order (episode_<k>); rename to env_<row> and drop the frames frozen envs
    keep recording after they finish (episode_remaining_*)."""
    for k, episode_seed in enumerate(np.asarray(metrics["seeds"])):
        (out / f"episode_{k}.mp4").rename(out / f"env_{int(episode_seed) - seed}.mp4")
    for leftover in out.glob("episode_remaining_*.mp4"):
        leftover.unlink()


def write_outputs(out: Path, cfg: DictConfig, metrics: dict, pairs: list[dict], success: np.ndarray,
                  final_distance: np.ndarray, min_distance: np.ndarray, wall_s: float) -> None:
    """config.json, results.txt (stable-worldmodel format) and episodes.csv (one row per pair)."""
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
        writer.writerow(["row", "start_x", "start_y", "goal_x", "goal_y", "route_cells", "success",
                         "final_distance", "min_distance"])
        for i, pair in enumerate(pairs):
            writer.writerow([i, *np.round(pair["start_xy"], 3), *np.round(pair["goal_xy"], 3), pair["route_cells"],
                             bool(success[i]), round(float(final_distance[i]), 3), round(float(min_distance[i]), 3)])


@hydra.main(version_base="1.3", config_path="../../configs/eval", config_name="pointmaze_harness")
def main(cfg: DictConfig):
    eval_cfg = cfg.eval
    pairs = draw_pairs(PointMazeHarnessEnv(task=cfg.world.task), eval_cfg.num_eval, eval_cfg.min_route_cells,
                       cfg.seed)
    options = [{"start_xy": p["start_xy"], "goal_xy": p["goal_xy"]} for p in pairs]
    routes = [p["route_cells"] for p in pairs]
    print(f"[harness] {len(pairs)} pairs, routes {min(routes)}..{max(routes)} cells", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_cfg = load_lewam(cfg.policy, which=cfg.seq_which)
    model = model.to(device).eval().requires_grad_(False)
    transform = {"pixels": img_transform(eval_cfg.img_size), "goal": img_transform(eval_cfg.img_size)}
    policy = build_policy(eval_cfg, model, model_cfg, cfg.seed, transform, goal_offsets=None)
    print(f"[eval] {type(policy).__name__}, {eval_cfg.horizon_schedule} schedule, h0={eval_cfg.h0}, "
          f"deadline {eval_cfg.goal_offset_steps} env steps, budget {eval_cfg.eval_budget}", flush=True)

    out = Path(cfg.output_dir) if cfg.output_dir else LW / "data" / "paper_repro" / "pointmaze_harness" / "scratch"
    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.set_policy(policy)
    start_time = time.time()
    metrics = world.evaluate(episodes=len(options), seed=cfg.seed, options=options, reset_mode="wait",
                             video=out if eval_cfg.save_video else None)
    wall_s = time.time() - start_time
    success = episode_successes(metrics, cfg.seed, len(options))
    final_distance, min_distance = episode_distances(world)
    if eval_cfg.save_video:
        rename_videos(out, metrics, cfg.seed)
    print(f"==== RESULTS ====\nsuccess_rate {metrics['success_rate']}")
    write_outputs(out, cfg, metrics, pairs, success, final_distance, min_distance, wall_s)
    print("wrote", out)


if __name__ == "__main__":
    main()
