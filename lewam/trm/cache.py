"""Offline latent cache: encode every frame of a set of episodes once through the frozen LeWAM encoder.

Pair sampling then becomes numpy fancy-indexing over an `(N, D)` array, and
every ablation head trains in seconds instead of re-running the encoder.

The cached latent is `LeWAM.encode(pixels)`, the same call `LeWAMPolicy._enc`
makes for history and goal frames at plan time, after the eval transform
(`lewam.eval.episodes.img_transform`). `.eval()` is load-bearing: the encoder
centre-crops in eval mode and random-crops in train mode, and its projector
carries `BatchNorm1d`.

Each cache is `<name>.npz` (the arrays) plus a `<name>.json` sidecar recording
which encoder wrote it -- see `load_cache` for why that check is not optional.

Ported from `playground/stable-worldmodel/src/trm/cache.py`. What changed: the
model is a LeWAM checkpoint (`load_lewam`), frames are read straight from the
h5 by `ep_offset` / `ep_len` instead of a swm dataset, and `--n-episodes`
caches a fixed random episode subset.

    python -m lewam.trm.cache --dataset puzzle_expert.h5 --checkpoint lewam-puzzle \\
        --n-episodes 5000 --verify
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  the swm h5 files compress frames with plugin filters
import numpy as np
import torch
from loguru import logger as logging
from stable_worldmodel.data.utils import get_cache_dir
from tqdm import tqdm

# Ground-truth position columns, kept alongside the latents for diagnostics
# (li26's "decoded XY" comparison label). First match wins.
XY_COLUMNS = (('pos_agent', 2), ('state', 2))


@dataclass
class LatentCache:
    """Frozen-encoder latents for a set of episodes, in flat `(episode, step)` order.

    `lengths`/`offsets` are the episode table over that flat row space, which is
    what `lewam.trm.pairs.build_segments` starts from.
    """

    emb: np.ndarray  # (N, D) float32
    episode_idx: np.ndarray  # (N,) int64, the dataset's episode id
    step_idx: np.ndarray  # (N,) int64
    lengths: np.ndarray  # (E,) int64
    offsets: np.ndarray  # (E,) int64
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.emb.shape[0]

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    @property
    def n_episodes(self) -> int:
        return self.lengths.shape[0]


def episode_table(lengths) -> tuple[np.ndarray, np.ndarray]:
    """`lengths` -> `(lengths, offsets)` as int64 over the flat row space."""
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    return lengths, offsets


def default_cache_path(dataset_name: str, checkpoint: str, cache_dir=None) -> Path:
    """`<cache>/latents/<dataset>__<checkpoint>.npz`, with path separators flattened."""
    flat = lambda s: str(s).replace('/', '--').replace('.lance', '')
    root = Path(get_cache_dir(cache_dir, sub_folder='latents'))
    return root / f'{flat(dataset_name)}__{flat(checkpoint)}.npz'


def select_episodes(n_total: int, n_episodes: int | None, seed: int) -> np.ndarray:
    """Sorted ids of a fixed random subset of episodes (all of them when n_episodes is None)."""
    if n_episodes is None or n_episodes >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_episodes, replace=False)).astype(np.int64)


def frames_to_pixels(frames: np.ndarray, transform, device) -> torch.Tensor:
    """(B, H, W, 3) uint8 frames -> (B, 3, H, W) float encoder input, on device."""
    pixels = torch.from_numpy(frames).to(device, non_blocking=True).permute(0, 3, 1, 2)
    return transform(pixels)


@torch.no_grad()
def build_latent_cache(
    h5: h5py.File, model, episode_ids: np.ndarray, transform, *, device, batch_size=256, meta=None
) -> LatentCache:
    """Encode every frame of the given episodes through *model*, in episode order."""
    model = model.to(device).eval()
    model.requires_grad_(False)
    ep_offset, ep_len = h5['ep_offset'][:], h5['ep_len'][:]
    lengths, offsets = episode_table(ep_len[episode_ids])
    n_rows = int(lengths.sum())

    emb = np.empty((n_rows, model.z_dim), dtype=np.float32)
    row = 0

    def encode_into(frames: np.ndarray) -> None:
        nonlocal row
        z = model.encode(frames_to_pixels(frames, transform, device))
        emb[row:row + len(frames)] = z.float().cpu().numpy()
        row += len(frames)

    pending = np.empty((0, *h5['pixels'].shape[1:]), dtype=np.uint8)   # batches cross episode boundaries
    for ep in tqdm(episode_ids, desc='encoding'):
        start = int(ep_offset[ep])
        pending = np.concatenate([pending, h5['pixels'][start:start + int(ep_len[ep])]])
        while len(pending) >= batch_size:
            encode_into(pending[:batch_size])
            pending = pending[batch_size:]
    if len(pending):
        encode_into(pending)
    assert row == n_rows, (row, n_rows)

    columns = {}
    keys = set(h5.keys())
    for name, width in XY_COLUMNS:
        if name in keys:
            columns['xy'] = np.concatenate(
                [h5[name][int(ep_offset[ep]):int(ep_offset[ep] + ep_len[ep]), :width] for ep in episode_ids]
            ).astype(np.float32)
            break

    return LatentCache(
        emb=emb,
        episode_idx=np.repeat(episode_ids, lengths),
        step_idx=np.concatenate([np.arange(n, dtype=np.int64) for n in lengths]),
        lengths=lengths,
        offsets=offsets,
        columns=columns,
        meta=dict(meta or {}, embed_dim=int(emb.shape[1]), n_rows=n_rows,
                  n_episodes=int(len(episode_ids)), episode_ids=episode_ids.tolist()),
    )


def save_cache(cache: LatentCache, path) -> Path:
    """Write the `.npz` plus its JSON sidecar; returns the npz path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        'emb': cache.emb,
        'episode_idx': cache.episode_idx,
        'step_idx': cache.step_idx,
        'lengths': cache.lengths,
        'offsets': cache.offsets,
    }
    arrays.update({f'col__{k}': v for k, v in cache.columns.items()})
    np.savez(path, **arrays)
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(cache.meta, f, indent=2)
    logging.info(f'latent cache written to {path} ({cache.emb.shape})')
    return path


def load_cache(path, expect_checkpoint=None, expect_dataset=None) -> LatentCache:
    """Load a cache, refusing one built from a different checkpoint or dataset.

    `checkpoint` is the frozen *world model* that encoded these frames, not the
    TRM head's own checkpoint. A cached latent means nothing except relative to
    the encoder that produced it; mixing two raises no error and changes no
    shape -- the head just ranks vectors from two coordinate systems. Hence a
    hard failure rather than a warning; the sidecar exists to make it cheap.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'latent cache not found: {path}')
    sidecar = path.with_suffix('.json')
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    for key, expected in (('checkpoint', expect_checkpoint), ('dataset', expect_dataset)):
        if expected is None:
            continue
        found = meta.get(key)
        if found != expected:
            raise ValueError(
                f"stale latent cache {path}: built with {key}='{found}', "
                f"but this run uses '{expected}'. Rebuild it with lewam.trm.cache."
            )

    with np.load(path) as data:
        columns = {k[len('col__'):]: data[k] for k in data.files if k.startswith('col__')}
        return LatentCache(
            emb=data['emb'],
            episode_idx=data['episode_idx'],
            step_idx=data['step_idx'],
            lengths=data['lengths'],
            offsets=data['offsets'],
            columns=columns,
            meta=meta,
        )


@torch.no_grad()
def verify_cache(
    cache: LatentCache, h5: h5py.File, model, transform, device, n_rows=8, tol=1e-4, seed=0, batch_size=256
) -> float:
    """Re-encode a few frames through the eval path (per-frame CPU transform) and compare to the cache.

    Padded up to `batch_size` so the encoder sees the build batch size: kernels
    can pick a different reduction order for another batch size, which is benign
    drift, not a bug. The per-frame transform is what `LeWAMPolicy` applies at
    eval, so this also checks the batched GPU transform of the build.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(cache), size=min(n_rows, len(cache)), replace=False)
    ep_offset = h5['ep_offset'][:]
    model = model.to(device).eval()

    pixels = torch.stack([
        transform(h5['pixels'][int(ep_offset[cache.episode_idx[r]] + cache.step_idx[r])]) for r in rows
    ])
    if pixels.shape[0] < batch_size:
        pad = pixels[:1].expand(batch_size - pixels.shape[0], *pixels.shape[1:])
        pixels = torch.cat([pixels, pad], dim=0)
    z = model.encode(pixels.to(device, torch.float32)).float().cpu().numpy()[: len(rows)]

    dev = float(np.abs(z - cache.emb[rows]).max())
    if dev > tol:
        raise AssertionError(f'cache mismatch: max|re-encoded - cached| = {dev:.3g} > {tol:g}')
    logging.info(f'cache fidelity: max deviation {dev:.3g} over {len(rows)} rows')
    return dev


def main(argv=None) -> None:
    from lewam.eval.episodes import img_transform
    from lewam.eval.loaders import load_lewam

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True, help='h5 file name under <cache>/datasets/')
    p.add_argument('--checkpoint', required=True, help='LeWAM run name under <cache>/checkpoints/')
    p.add_argument('--which', default='best', help='lewam_<which>.pt')
    p.add_argument('--n-episodes', type=int, default=None, help='random episode subset (default: all)')
    p.add_argument('--episode-seed', type=int, default=0)
    p.add_argument('--out', default=None, help='output .npz (default: <cache>/latents/...)')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--verify', action='store_true', help='re-encode a few rows and compare')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)

    out = Path(args.out) if args.out else default_cache_path(args.dataset, args.checkpoint, args.cache_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f'{out} exists; pass --overwrite to rebuild')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_lewam(args.checkpoint, which=args.which)
    transform = img_transform(args.img_size)
    h5_path = Path(get_cache_dir(args.cache_dir, sub_folder='datasets')) / args.dataset

    with h5py.File(h5_path, 'r') as h5:
        episode_ids = select_episodes(len(h5['ep_len']), args.n_episodes, args.episode_seed)
        logging.info(f'{h5_path}: {len(episode_ids)} of {len(h5["ep_len"])} episodes, '
                     f'{int(h5["ep_len"][:][episode_ids].sum())} frames')
        cache = build_latent_cache(
            h5, model, episode_ids, transform, device=device, batch_size=args.batch_size,
            meta={'dataset': args.dataset, 'checkpoint': args.checkpoint, 'which': args.which,
                  'img_size': args.img_size, 'episode_seed': args.episode_seed},
        )
        save_cache(cache, out)
        if args.verify:
            verify_cache(cache, h5, model, transform, device, batch_size=args.batch_size)


__all__ = [
    'LatentCache',
    'build_latent_cache',
    'default_cache_path',
    'episode_table',
    'load_cache',
    'save_cache',
    'select_episodes',
    'verify_cache',
]


if __name__ == '__main__':
    main()
