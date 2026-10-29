"""PointMaze: reach OGBench's red goal marker (xy = 0, 0) from the start of a dataset episode.

Episodes are those whose agent passes within REDDOT_RADIUS of the marker at least MIN_OFFSET steps in; the goal
frame is the closest approach on that first visit, the start is step 0 of the episode. Runs eval_lewam.main with
eval.full_traj=true (per-env goal offset and budget 2 x offset) and only the episode draw replaced.

  python scripts/paper_repro/eval_lewam_reddot.py --config-name pointmaze policy=lewam-pointmaze-large \
      seed=42 eval.full_traj=true output_dir=data/paper_repro/pointmaze_reddot_max/reactive/42
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_lewam  # noqa: E402
from lewam.eval.episodes import episode_col  # noqa: E402

REDDOT_RADIUS = 0.25  # goal frame visibly on the marker (OGBench's success tolerance is 0.5)
MIN_OFFSET = 50      # the paper's PointMaze goal offset; shorter first visits are left out


def sample_reddot_episodes(dataset, num_eval, seed, goal_offset_steps, full_traj=False, random_start=True,
                           min_episode_len=0):
    """Up to num_eval episodes (all if fewer qualify), drawn from seed; starts at step 0, goal at the red dot."""
    assert full_traj, "run with eval.full_traj=true: per-env goal offsets and budgets"
    row_episodes = np.asarray(dataset.get_col_data(episode_col(dataset))).reshape(-1)
    row_steps = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    row_xy = np.asarray(dataset.get_col_data("state"))[:, :2]
    distance = np.linalg.norm(row_xy, axis=1)

    goal_step_of = {}
    for episode in np.unique(row_episodes[distance < REDDOT_RADIUS]):
        rows = np.nonzero(row_episodes == episode)[0]
        rows = rows[np.argsort(row_steps[rows])]
        near = distance[rows] < REDDOT_RADIUS
        first = int(np.argmax(near))
        last = first + (int(np.argmin(near[first:])) if not near[first:].all() else len(near) - first)
        closest = first + int(np.argmin(distance[rows[first:last]]))
        goal_step_of[int(episode)] = int(row_steps[rows[closest]])

    eligible = sorted(e for e, step in goal_step_of.items() if step >= MIN_OFFSET)
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(eligible, size=min(num_eval, len(eligible)), replace=False))
    print(f"[reddot] {len(eligible)} eligible episodes, {len(picks)} drawn; goal offsets "
          f"{min(goal_step_of[e] for e in picks)}..{max(goal_step_of[e] for e in picks)}", flush=True)
    return picks.tolist(), [0] * len(picks), [goal_step_of[e] for e in picks]


if __name__ == "__main__":
    eval_lewam.sample_eval_episodes = sample_reddot_episodes
    eval_lewam.main()
