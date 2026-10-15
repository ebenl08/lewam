"""P1-M3 identity checks: PushT start/goal draws and success predicate, LeWAM (swm 0.1.1) vs stable-worldmodel fork.

Each side runs in its own conda env and writes JSON to data/planner_comparison/p1/checks/; --side compare diffs them.
- draws: 25 seeds x 30 episodes, offset 25, on pusht_expert_train.h5: LW sample_eval_episodes vs PG build_eval_indices.
- predicate: PushT eval_state on 1,000 (state, state 25 steps later) pairs + 1,000 random cross pairs from the dataset.

CLI (from playground/lewam/):
    /home/eben/.conda/envs/lewam/bin/python scripts/planner_comparison/check_identity.py --side lw
    /home/eben/.conda/envs/swm/bin/python scripts/planner_comparison/check_identity.py --side pg
    python scripts/planner_comparison/check_identity.py --side compare
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PG_SRC = REPO.parent / 'stable-worldmodel' / 'src'
CHECKS = REPO / 'data' / 'planner_comparison' / 'p1' / 'checks'
DATASET = 'pusht_expert_train'
H5 = Path.home() / '.stable_worldmodel' / 'datasets' / f'{DATASET}.h5'
NUM_EVAL, OFFSET, NUM_PAIRS = 30, 25, 1000
sys.path.insert(0, str(REPO / 'scripts' / 'planner_comparison'))
from make_jobs import SEEDS  # noqa: E402


def predicate_pairs():
    """(cur, goal) state pairs: same-episode pairs OFFSET steps apart, then random cross pairs. Fixed by rng(0)."""
    with h5py.File(H5, 'r') as f:
        states = f['state'][:]
        episodes = f['episode_idx'][:]
    rng = np.random.default_rng(0)
    rows = rng.choice(len(states) - OFFSET, size=4 * NUM_PAIRS, replace=False)
    rows = rows[episodes[rows] == episodes[rows + OFFSET]][:NUM_PAIRS]
    cross_a, cross_b = rng.choice(len(states), size=(2, NUM_PAIRS))
    cur = np.concatenate([states[rows], states[cross_a]])
    goal = np.concatenate([states[rows + OFFSET], states[cross_b]])
    return cur, goal


def predicate(cur, goal):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers swm/PushT-v1

    env = gym.make('swm/PushT-v1').unwrapped
    env.reset(seed=0)
    return [bool(env.eval_state(g, c)[0]) for c, g in zip(cur, goal)]


def draws_lw():
    from lewam.eval.episodes import get_dataset, sample_eval_episodes

    dataset = get_dataset(DATASET, [])
    out = {}
    for seed in SEEDS:
        episodes, starts, _ = sample_eval_episodes(dataset, NUM_EVAL, seed, OFFSET)
        out[seed] = [list(map(int, episodes)), list(map(int, starts))]
    return out


def draws_pg():
    from omegaconf import OmegaConf
    import stable_worldmodel as swm

    sys.path.insert(0, str(PG_SRC))
    from eval import build_eval_indices

    dataset = swm.data.load_dataset(f'{DATASET}.h5', keys_to_cache=[])  # as the PG runs: eval.dataset_name=pusht_expert_train.h5
    eval_cfg = OmegaConf.create({'num_eval': NUM_EVAL, 'goal_offset_steps': OFFSET})
    out = {}
    for seed in SEEDS:
        episodes, starts = build_eval_indices(dataset, eval_cfg, seed=seed)
        out[seed] = [list(map(int, episodes)), list(map(int, starts))]
    return out


def compare():
    lw, pg = (json.loads((CHECKS / f'identity_{side}.json').read_text()) for side in ('lw', 'pg'))
    lines = []
    same_draws = [seed for seed in lw['draws'] if lw['draws'][seed] == pg['draws'][seed]]
    lines.append(f'draws: {len(same_draws)}/{len(lw["draws"])} seeds identical (episode, start) lists')
    a, b = np.array(lw['predicate']), np.array(pg['predicate'])
    for name, part in (('offset-25 pairs', slice(0, NUM_PAIRS)), ('random cross pairs', slice(NUM_PAIRS, None))):
        lines.append(f'predicate {name}: {int((a[part] == b[part]).sum())}/{len(a[part])} agree; '
                     f'success lw {int(a[part].sum())}, pg {int(b[part].sum())}')
    cur, goal = predicate_pairs()
    diff = np.abs(goal[:, 4] - cur[:, 4])
    lines.append(f'|angle diff| > 2 pi in {int((diff > 2 * np.pi).sum())} of {len(diff)} pairs; '
                 f'dataset angle range [{cur[:, 4].min():.2f}, {cur[:, 4].max():.2f}]')
    text = '\n'.join(lines)
    (CHECKS / 'identity_summary.txt').write_text(text + '\n')
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', choices=('lw', 'pg', 'compare'), required=True)
    args = parser.parse_args()
    CHECKS.mkdir(parents=True, exist_ok=True)
    if args.side == 'compare':
        compare()
        return
    draws = draws_lw() if args.side == 'lw' else draws_pg()
    result = dict(draws=draws, predicate=predicate(*predicate_pairs()))
    (CHECKS / f'identity_{args.side}.json').write_text(json.dumps(result))
    print('wrote', CHECKS / f'identity_{args.side}.json')


if __name__ == '__main__':
    main()
