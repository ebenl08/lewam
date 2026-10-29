"""Goal-anchored episode draw: the goal is each episode's last frame."""

import numpy as np
import pytest

from lewam.eval.episodes import sample_goal_anchored_episodes


class _FakeDataset:
    """Per-step episode_idx and step_idx columns for episodes of the given lengths."""

    column_names = ["episode_idx", "step_idx"]

    def __init__(self, lengths):
        self.columns = {
            "episode_idx": np.concatenate([np.full(length, episode) for episode, length in enumerate(lengths)]),
            "step_idx": np.concatenate([np.arange(length) for length in lengths]),
        }

    def get_col_data(self, name):
        return self.columns[name]


def test_start_is_offset_before_last_frame():
    dataset = _FakeDataset([87, 40, 120, 51])
    episodes, starts, goal_offsets = sample_goal_anchored_episodes(dataset, 3, seed=0, goal_offset_steps=50)
    assert episodes == [0, 2, 3]
    assert starts == [36, 69, 0]
    assert goal_offsets is None


def test_draw_is_seeded_and_sorted():
    dataset = _FakeDataset([60 + episode for episode in range(20)])
    first = sample_goal_anchored_episodes(dataset, 5, seed=8902, goal_offset_steps=50)
    second = sample_goal_anchored_episodes(dataset, 5, seed=8902, goal_offset_steps=50)
    assert first == second
    assert first[0] == sorted(first[0])
    assert len(set(first[0])) == 5


def test_too_few_long_episodes():
    dataset = _FakeDataset([87, 40, 120, 51])
    with pytest.raises(ValueError, match="3 episodes longer than 50"):
        sample_goal_anchored_episodes(dataset, 4, seed=0, goal_offset_steps=50)
