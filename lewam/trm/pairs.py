"""Pair sampling: segments, the three delta distributions, labels, the control.

This is the part that is easy to mistake for an implementation detail. li26's
Table 16 holds architecture and pair budget fixed and varies only P(delta):
full-episode balanced reaches 97.5%, capping at delta <= 50 reaches 35.0% on the
same budget. The metric has to be trained at the temporal scale it will be used
at, and the terminal selector ranks candidates a full planning horizon away from
the goal.

See the design note, section 4, including the worked [500, 500, 120, 60] example
that tests/test_trm_pairs.py uses as a fixture. Ported verbatim from
`playground/stable-worldmodel/src/trm/pairs.py`.

Taxonomy of segments, pairs and deltas:

segment 0 (episode 0, rows 0..499)
  ├─ pair (row 12, row 340)   delta=328
  ├─ pair (row 5,  row 9)     delta=4
  └─ pair (row 200, row 201)  delta=1
segment 1 (episode 1, rows 500..999)
  ├─ pair (row 510, row 990)  delta=480
  └─ ...
  
"""

import numpy as np

MAX_REJECTION_ROUNDS = 64


def _seg_p(valid_delta , length_weighted):
    """Per-segment sampling probability over the given subset.

    None means uniform (numpy's default), matching the paper's "sample an
    episode". length_weighted makes it proportional to length -- i.e. uniform
    over *rows* rather than over episodes, which over-samples long segments.
    """
    if not length_weighted:
        return None
    valid_delta  = np.asarray(valid_delta , dtype=np.float64)
    return valid_delta  / valid_delta .sum()


def build_segments(
    episode_idx,
    goal_met=None,
    split_on_goal_reset=False,
    min_length=2,
):
    """Pair-sampling units, as (lengths, offsets) over the flat row space.

    One segment per episode by default. split_on_goal_reset additionally cuts at
    every goal_met. This is to prevent the jump in the latent space when the input emb
    sees a new goal frame

    Segments shorter than min_length are DROPPED: they cannot furnish a pair.
    Their rows are then never sampled, which is why the returned table carries
    explicit offsets rather than assuming the segments tile the rows.
    """
    episode_idx = np.asarray(episode_idx).reshape(-1)
    n_rows = episode_idx.shape[0]
    if n_rows == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    
    # ------------------------------------------------
    # [0, 0, 1] -diff-> [0 1] --flatnonzero--> [1]
    # [1] -(+1)-> [2], 2 is indeed the start of second episode
    # ------------------------------------------------
    cuts = list(np.flatnonzero(np.diff(episode_idx) != 0) + 1)

    if split_on_goal_reset:
        if goal_met is None:
            raise ValueError('split_on_goal_reset=True requires goal_met')
        goal_met = np.asarray(goal_met).reshape(-1).astype(bool)
        cuts += list(np.flatnonzero(goal_met) + 1)

    starts = np.unique(np.array([0, *cuts, n_rows], dtype=np.int64))
    offsets, lengths = starts[:-1], np.diff(starts)
    keep = lengths >= min_length
    return lengths[keep].astype(np.int64), offsets[keep].astype(np.int64)


def split_segments(n_segments, val_frac, rng):
    """Hold out whole SEGMENTS, not pairs; returns (train_idx, val_idx).

    """
    perm = rng.permutation(n_segments)
    n_val = int(round(val_frac * n_segments))
    if val_frac > 0 and n_val == 0:
        n_val = 1  # a nonzero request must not silently produce an empty split
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def _random_rows(lengths, n_pairs, rng, min_delta, max_delta, length_weighted):
    """Draw i and j independently inside one segment, then filter by delta.

    Returns (seg, delta); the caller redraws the start uniformly given delta.
    That redraw leaves the joint distribution unchanged -- still uniform over
    unordered row pairs within a segment -- while letting all three modes share
    one tail. Rejection is done in over-drawn rounds rather than one pair at a
    time, and a budget that cannot be met raises rather than silently returning
    a short table.
    """
    lengths = np.asarray(lengths)
    valid_delta  = lengths - 1
    elig = np.flatnonzero(valid_delta  >= min_delta)
    if elig.size == 0:
        raise ValueError(f'no segment can realize delta >= {min_delta}')

    segs, deltas, have = [], [], 0
    for _ in range(MAX_REJECTION_ROUNDS):
        if have >= n_pairs:
            break
        # 64 is just so that the # of drawn samples are never too small, even if only little is needed
        # 1.5 is to overestimate how much you need as a buffer
        n_draw = int((n_pairs - have) * 1.5) + 64
        s = rng.choice(
            elig, size=n_draw, replace=True, p=_seg_p(valid_delta [elig], length_weighted)
        )
        span = lengths[s]
        i = (rng.random(n_draw) * span).astype(np.int64)
        j = (rng.random(n_draw) * span).astype(np.int64)
        d = np.abs(i - j)

        keep = d >= min_delta
        if max_delta is not None:
            keep &= d <= max_delta
        segs.append(s[keep])
        deltas.append(d[keep])
        have += int(keep.sum())
    else:
        raise RuntimeError(
            f'random_rows: only {have}/{n_pairs} pairs survived the delta filter after '
            f'{MAX_REJECTION_ROUNDS} rounds -- min_delta/max_delta are too narrow for '
            'these segment lengths'
        )

    return np.concatenate(segs)[:n_pairs], np.concatenate(deltas)[:n_pairs]


def sample_pairs(
    lengths,
    offsets,
    n_pairs,
    rng,
    *,
    mode='balanced',
    n_bins=10,
    min_delta=1,
    max_delta=None,
    length_weighted=False,
    flip=True,
):
    """Draw n_pairs (row_i, row_j, delta) triples, each pair inside one segment.

    lengths/offsets are the segment table from build_segments: by default
    exactly the dataset's own episode lengths and start rows in the flat table.

    Three modes, differing ONLY in the distribution over temporal separation
    delta -- which is the whole ballgame, since the terminal selector ranks
    candidates a full planning horizon away from the goal, and a metric trained
    on short gaps has never seen that regime.

      'random_rows'   i and j drawn independently inside a segment, so
                      P(delta) ~ (L - delta): TRIANGULAR, mass piled on short
                      gaps. The naive implementation, and the failure mode the
                      paper's short-horizon ablation isolates.
                      
      'uniform_delta' delta drawn first, uniform over what the segment can
                      realize, then a valid start. Flat over delta WITHIN a
                      segment.
                      
      'balanced'      delta stratified into n_bins equal-count buckets; a bucket
                      is filled only from segments long enough to realize it.
                      Flat and equal-count over delta GLOBALLY -- which is what
                      'uniform_delta' stops being the moment segment lengths
                      differ, since short segments can only ever emit small
                      deltas. Default.

    max_delta caps the horizon (the paper's max-delta=50 ablation).
    """
    lengths, offsets = np.asarray(lengths), np.asarray(offsets)
    if lengths.shape != offsets.shape:
        raise ValueError(f'lengths {lengths.shape} and offsets {offsets.shape} disagree')
    if lengths.size == 0:
        raise ValueError('empty segment table')
    if min_delta < 1:
        raise ValueError(f'min_delta must be >= 1, got {min_delta}')

    valid_delta  = lengths - 1  # largest delta each segment can realize # -1 to enforce delta >= 1
    dmax = int(valid_delta .max() if max_delta is None else min(max_delta, valid_delta .max()))
    if dmax < min_delta:
        raise ValueError(f'no segment can realize delta >= {min_delta}')

    if mode == 'balanced':
        # n_bins buckets need n_bins+1 fenceposts; bucket k is
        # [edges[k], edges[k+1]). The upper endpoint is dmax+1 because the
        # buckets are half-open, so the last one must reach past dmax to
        # include it.
        edges = np.unique(
            np.linspace(min_delta, dmax + 1, n_bins + 1).round().astype(np.int64)
        )
        # Split n_pairs into len(edges)-1 parts summing to exactly n_pairs, even
        # when it does not divide evenly (which integer division would not).
        quota = np.diff(np.linspace(0, n_pairs, len(edges)).round().astype(np.int64))
        segs, deltas = [], []
        for lo, hi, n in zip(edges[:-1], edges[1:], quota):
            
            # eligible -> indices of segments that fulfills the min_delta requirement
            elig = np.flatnonzero(valid_delta  >= lo)
            if n == 0 or elig.size == 0: continue
            s = rng.choice(
                elig, size=n, replace=True, p=_seg_p(valid_delta [elig], length_weighted)
            )
            hi_s = np.minimum(hi - 1, valid_delta [s])  # per-segment cap
            d = lo + (rng.random(n) * (hi_s - lo + 1)).astype(np.int64)
            segs.append(s)
            deltas.append(d)
        if not segs:
            raise ValueError('balanced: no bucket could be filled')
        seg, delta = np.concatenate(segs), np.concatenate(deltas)

    elif mode == 'uniform_delta':
        elig = np.flatnonzero(valid_delta >= min_delta)
        seg = rng.choice(
            elig, 
            size=n_pairs, 
            replace=True,
            p=_seg_p(valid_delta[elig], length_weighted)
        )# length_weighted are boolean val
        hi_s = np.minimum(dmax, valid_delta[seg])
        delta = min_delta + (rng.random(n_pairs) * (hi_s - min_delta + 1)).astype(np.int64)

    elif mode == 'random_rows':
        seg, delta = _random_rows(lengths, n_pairs, rng, min_delta, max_delta, length_weighted)

    else:
        raise ValueError(f'unknown mode {mode!r}; expected balanced/uniform_delta/random_rows')

    # start time, uniform over the window that leaves room for delta
    t = (rng.random(seg.size) * (lengths[seg] - delta)).astype(np.int64)
    i = offsets[seg] + t
    j = i + delta

    if flip:
        # The label is symmetric, the feature map is not. li26 does this ("use
        # rows (e,t) and (e,t+Delta) in random order") but never says why and
        # never ablates it. Note this flip -- NOT the abs() in the label -- is
        # where symmetry is ASSERTED: it presents (z_j, z_i) carrying the
        # forward-traversal label, claiming the reverse costs the same. Kept
        # switchable; see the design note, "The ordering ablation".
        swap = rng.random(seg.size) < 0.5
        i, j = np.where(swap, j, i), np.where(swap, i, j)

    return i.astype(np.int64), j.astype(np.int64), delta.astype(np.int64)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _temporal_label(cache, i, j, delta):
    """y = |t_i - t_j|, straight from the sampler.

    Generic and oracle-free: recoverable from the log alone, with no simulator
    state. A small gap means the two states are connected by few feasible
    actions under the logged dynamics, which is the reachability proxy.
    """
    return delta.astype(np.float32)


def _xy_label(cache, i, j, delta):
    """y = ||xy_i - xy_j||, from the dataset's ground-truth column.

    Equals the true geodesic in this convex arena -- BoundaryWalls fences it and
    GoalLevel2 adds only non-colliding hazards and pushable vases, so nothing
    statically obstructs the agent. li26's "decoded XY" comparison point, and
    NOT oracle-free: it reads privileged simulator state.
    """
    if 'xy' not in cache.columns:
        raise KeyError("label 'xy' needs the xy column; rebuild the cache with it")
    xy = np.asarray(cache.columns['xy'], dtype=np.float64)
    return np.linalg.norm(xy[i] - xy[j], axis=-1).astype(np.float32)


LABELS = {
    'temporal': _temporal_label,  # default
    'xy': _xy_label,
}

def shuffle_labels(y, rng):
    """Permute labels -- the negative control. TRAIN labels only.
    
    In the paper this drops hard TwoRoom success from 99.0% to 0.0% across six runs, which is what licenses
    the claim that temporal structure -- not "any learned scalar cost" -- does
    the work. One line, and not optional.
    """
    return rng.permutation(y)

def segment_of(offsets, rows):
    """Which segment each flat row index falls in. Diagnostics and tests."""
    return np.searchsorted(np.asarray(offsets), np.asarray(rows), side='right') - 1

def delta_histogram(delta, bins=10, dmax=None):
    """(counts, edges) over delta -- the diagnostic that makes the modes visible.
    """
    dmax = int(np.max(delta)) if dmax is None else int(dmax)
    edges = np.linspace(1, dmax + 1, bins + 1)
    counts, edges = np.histogram(delta, bins=edges)
    return counts, edges


__all__ = [
    'LABELS',
    'build_segments',
    'delta_histogram',
    'sample_pairs',
    'segment_of',
    'shuffle_labels',
    'split_segments',
]
