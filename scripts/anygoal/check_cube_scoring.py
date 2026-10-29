"""Expert-replay check of the Cube-variant scoring: replay each episode's dataset actions and score the run
with swm's terminations (as eval_lewam.py does) and with AnyGoal's Cube metrics (anygoal/planning.py).

  python scripts/anygoal/check_cube_scoring.py --env cube_hop --seed 8902 --num-eval 30 \
      --out data/beam_smoke/analysis/cube_scoring_check
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registers the HDF5 format
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

LW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LW / "scripts"))
from eval_lewam import register_env  # noqa: E402

from lewam.eval.episodes import get_dataset, sample_eval_episodes  # noqa: E402


class ReplayPolicy(swm.policy.BasePolicy):
    """Execute each episode's dataset actions from its start step, then hold still."""

    def __init__(self, actions: np.ndarray):
        super().__init__()
        self.type = "replay"
        self.actions = actions   # (num_envs, goal_offset_steps, action_dim)
        self.t = 0

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        if self.t < self.actions.shape[1]:
            action = self.actions[:, self.t]
        else:
            action = np.zeros_like(self.actions[:, 0])
        self.t += 1
        return action


def replay(cfg, dataset, episodes, starts, actions, anygoal_metrics: bool, video: Path | None = None) -> dict:
    """One evaluate call on a fresh World with the replay policy; video: folder for one mp4 per env."""
    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.set_policy(ReplayPolicy(actions))
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    if anygoal_metrics:
        from anygoal.envs.cube_hop.env import prepare_cube_evaluation

        callables = prepare_cube_evaluation(world, callables)
    metrics = world.evaluate(
        dataset=dataset, start_steps=starts, goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget, episodes_idx=episodes, callables=callables, video=video,
    )
    if anygoal_metrics:
        from anygoal.envs.cube_hop.env import add_cube_evaluation_metrics

        metrics = add_cube_evaluation_metrics(world, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["cube_hop", "cube_diagonal"])
    parser.add_argument("--seed", type=int, default=8902)
    parser.add_argument("--num-eval", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true", help="save the AnyGoal-scored run as env_<i>.mp4")
    args = parser.parse_args()
    out = args.out / args.env
    out.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(LW / "configs" / "eval"), version_base="1.3"):
        cfg = compose(config_name=args.env, overrides=[f"eval.num_eval={args.num_eval}", "policy=replay"])
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    register_env(cfg.world.env_name)

    offset = cfg.eval.goal_offset_steps
    dataset = get_dataset(cfg.eval.dataset_name, list(cfg.dataset.keys_to_cache), cfg.cache_dir)
    episodes, starts, _ = sample_eval_episodes(dataset, args.num_eval, args.seed, offset)
    chunks = dataset.load_chunk(np.asarray(episodes), np.asarray(starts), np.asarray(starts) + offset + 1)
    actions = np.stack([np.asarray(chunk["action"])[:offset] for chunk in chunks]).astype(np.float32)
    assert np.isfinite(actions).all(), "non-finite dataset actions in the replay window"

    old = replay(cfg, dataset, episodes, starts, actions, anygoal_metrics=False)
    tag = f"seed{args.seed}_n{args.num_eval}"
    video = out / f"video_{tag}" if args.video else None
    new = replay(cfg, dataset, episodes, starts, actions, anygoal_metrics=True, video=video)

    errors = new["cube_final_errors"]
    summary = {
        "env": args.env, "seed": args.seed, "episodes": episodes, "starts": starts,
        "goal_offset_steps": offset, "eval_budget": int(cfg.eval.eval_budget),
        "old_terminations": {"success_rate": float(old["success_rate"]),
                             "episode_successes": np.asarray(old["episode_successes"]).tolist()},
        "criterion_3": new["cube_hop_criterion_3"],
        "original_cube": new["original_cube_criterion"],
        "tolerances": new["cube_evaluation_tolerances"],
        "final_errors": errors,
    }
    (out / f"replay_{tag}.json").write_text(json.dumps(summary, indent=1, default=float))

    print(f"[{args.env}] seed {args.seed}, {args.num_eval} episodes, offset {offset}, budget {cfg.eval.eval_budget}")
    print(f"  old (terminations):       {old['success_rate']:.1f}%")
    print(f"  criterion 3 (any step):   {new['cube_hop_criterion_3']['success_rate']:.1f}%")
    print(f"  original cube (any step): {new['original_cube_criterion']['success_rate']:.1f}%")
    for name, values in errors.items():
        print(f"  final {name} error: median {np.median(values):.4f}, max {np.max(values):.4f}")
    print("wrote", out / f"replay_{tag}.json")


if __name__ == "__main__":
    main()
