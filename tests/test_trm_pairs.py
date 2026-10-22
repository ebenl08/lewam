"""Sampler assertions: no model in the loop, pure numpy, runs in seconds.

Verification 2 of the design note. These run BEFORE any GPU time is spent,
because a sampler that quietly violates its contract invalidates every number
downstream. The [500, 500, 120, 60] fixture is the note's own worked example.
"""

import numpy as np
import pytest

from lewam.trm.pairs import (
    LABELS,
    build_segments,
    delta_histogram,
    sample_pairs,
    segment_of,
    shuffle_labels,
    split_segments,
)

LENGTHS = np.array([500, 500, 120, 60])
OFFSETS = np.array([0, 500, 1000, 1120])
MODES = ['balanced', 'uniform_delta', 'random_rows']


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize('mode', MODES)
def test_pairs_stay_inside_one_segment(mode):
    i, j, d = sample_pairs(LENGTHS, OFFSETS, 2000, rng(), mode=mode)
    seg_i, seg_j = segment_of(OFFSETS, i), segment_of(OFFSETS, j)
    assert np.array_equal(seg_i, seg_j)
    lo, hi = OFFSETS[seg_i], OFFSETS[seg_i] + LENGTHS[seg_i]
    assert np.all((i >= lo) & (i < hi) & (j >= lo) & (j < hi))


@pytest.mark.parametrize('mode', MODES)
def test_delta_matches_row_gap(mode):
    i, j, d = sample_pairs(LENGTHS, OFFSETS, 2000, rng(), mode=mode)
    assert np.array_equal(np.abs(j - i), d)


@pytest.mark.parametrize('mode', MODES)
def test_delta_bounds_respected(mode):
    i, j, d = sample_pairs(
        LENGTHS, OFFSETS, 2000, rng(), mode=mode, min_delta=10, max_delta=50
    )
    assert d.min() >= 10 and d.max() <= 50


@pytest.mark.parametrize('mode', MODES)
def test_exact_pair_count(mode):
    i, j, d = sample_pairs(LENGTHS, OFFSETS, 997, rng(), mode=mode, n_bins=5)
    assert len(i) == len(j) == len(d) == 997


def test_balanced_quota_is_exact():
    """The note's worked example: 1000 pairs, 5 bins, 200 per delta-band."""
    _, _, d = sample_pairs(LENGTHS, OFFSETS, 1000, rng(), mode='balanced', n_bins=5)
    edges = np.array([1, 101, 201, 300, 400, 500])
    counts = [int(((d >= lo) & (d < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    assert counts == [200] * 5


def test_balanced_top_bin_only_uses_long_segments():
    """Segments 2 and 3 (usable 119, 59) cannot realize delta >= 400."""
    i, j, d = sample_pairs(LENGTHS, OFFSETS, 1000, rng(), mode='balanced', n_bins=5)
    top = segment_of(OFFSETS, np.minimum(i, j))[d >= 400]
    assert set(np.unique(top)) <= {0, 1}


def test_balanced_beats_uniform_delta_on_long_horizon_coverage():
    """uniform_delta stops being flat globally once segment lengths differ."""
    _, _, d_bal = sample_pairs(LENGTHS, OFFSETS, 20000, rng(), mode='balanced')
    _, _, d_uni = sample_pairs(LENGTHS, OFFSETS, 20000, rng(), mode='uniform_delta')
    assert (d_bal > 400).mean() > (d_uni > 400).mean()


def test_random_rows_is_triangular():
    """P(delta) ~ (L - delta) gives E[delta] ~ L/3, not L/2."""
    lengths, offsets = np.array([500] * 4), np.arange(4) * 500
    _, _, d = sample_pairs(lengths, offsets, 40000, rng(), mode='random_rows')
    assert 150 < d.mean() < 185  # ~167, vs ~250 for a flat distribution
    assert (d > 400).mean() < 0.06  # ~4%, vs ~20% under a flat distribution


def test_balanced_is_flat():
    _, _, d = sample_pairs(LENGTHS, OFFSETS, 20000, rng(), mode='balanced', n_bins=10)
    counts, _ = delta_histogram(d, bins=10, dmax=499)
    assert counts.max() / counts.min() < 1.15


def test_flip_off_gives_directed_pairs():
    i, j, _ = sample_pairs(LENGTHS, OFFSETS, 2000, rng(), flip=False)
    assert np.all(i < j)


def test_flip_on_is_balanced_between_orders():
    i, j, _ = sample_pairs(LENGTHS, OFFSETS, 20000, rng(), flip=True)
    assert 0.45 < (i > j).mean() < 0.55


def test_segments_from_episode_ids():
    ep = np.repeat([0, 1, 2, 3], LENGTHS)
    lengths, offsets = build_segments(ep)
    assert np.array_equal(lengths, LENGTHS) and np.array_equal(offsets, OFFSETS)


def test_goal_reset_splits_and_drops_short_segments():
    ep = np.repeat([0, 1], [10, 10])
    goal_met = np.zeros(20, bool)
    goal_met[4] = True  # episode 0 -> [0:5), [5:10)
    goal_met[18] = True  # episode 1 -> [10:19), [19:20) <- length 1, dropped
    lengths, offsets = build_segments(ep, goal_met=goal_met, split_on_goal_reset=True)
    assert np.array_equal(offsets, [0, 5, 10])
    assert np.array_equal(lengths, [5, 5, 9])


def test_split_holds_out_whole_segments():
    train, val = split_segments(100, 0.1, rng())
    assert len(val) == 10 and len(train) == 90
    assert not set(train.tolist()) & set(val.tolist())


def test_shuffle_labels_permutes():
    y = np.arange(1000, dtype=np.float32)
    ys = shuffle_labels(y, rng())
    assert np.array_equal(np.sort(ys), y) and not np.array_equal(ys, y)


def test_temporal_label_is_the_delta():
    i, j, d = sample_pairs(LENGTHS, OFFSETS, 100, rng())
    assert np.array_equal(LABELS['temporal'](None, i, j, d), d.astype(np.float32))


def test_impossible_budget_raises():
    with pytest.raises((ValueError, RuntimeError)):
        sample_pairs(
            LENGTHS, OFFSETS, 100, rng(), mode='random_rows', min_delta=499, max_delta=499
        )
