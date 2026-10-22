"""Trainer for the TRM head: cached LeWAM latents in, a ranking metric out.

The pair table is ~100k x 2 x 192 floats and lives entirely in GPU memory, so
there is no dataloader. 100k pairs at batch 1024 is 98 steps per epoch, i.e.
seconds per head -- which is what makes the ablation arms (shuffled labels,
flip on/off, the three sampler modes) cheap enough to actually run.

    python -m lewam.trm.train -cn train dataset=tworoom.h5 checkpoint=lewam-tworoom \\
        output_model_name=trm_tworoom_v1

label_scale is the hyperparameter of distance_scale mentioned in the paper.

Ported from `playground/stable-worldmodel/src/trm/train.py`. What changed:
checkpoints go to `<out_root>/<output_model_name>/` via `lewam.trm.checkpoint`
instead of `save_pretrained`; metrics go to `metrics.json` instead of wandb;
the Spearman warning threshold is `spearman_floor` instead of a hardcoded 0.9.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from scipy.stats import rankdata, spearmanr

from lewam.trm.cache import default_cache_path, load_cache
from lewam.trm.checkpoint import save_head
from lewam.trm.head import PairwiseDistanceHead
from lewam.trm.pairs import LABELS, build_segments, sample_pairs, shuffle_labels, split_segments

LW_ROOT = Path(__file__).resolve().parents[2]


def draw_pairs(cache, segments, cfg, n_pairs, rng, flip):
    """Sample pairs from a segment SUBSET and label them."""
    lengths, offsets = segments
    i, j, delta = sample_pairs(
        lengths,
        offsets,
        n_pairs,
        rng,
        mode=cfg.sampler.mode,
        n_bins=cfg.sampler.n_bins,
        min_delta=cfg.sampler.min_delta,
        max_delta=cfg.sampler.max_delta,
        length_weighted=cfg.sampler.length_weighted,
        flip=flip,
    )
    y = LABELS[cfg.label](cache, i, j, delta)
    return i, j, y


def pairs_to_device(emb, i, j, y, device):
    """Materialize (P, D) tensors for both sides of the pair table on device.

    100k pairs at D=192 is ~150 MB for both sides -- cheap enough to hold
    outright, which is what removes the dataloader from the loop.
    """
    z = torch.from_numpy(emb)
    return (
        z[torch.from_numpy(i)].to(device),
        z[torch.from_numpy(j)].to(device),
        torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device),
    )


def label_stats(name, y):
    logging.info(
        f'{name} labels: n={y.shape[0]} min={y.min():.0f} max={y.max():.0f} mean={y.mean():.1f} '
        f'std={y.std():.1f} median={np.percentile(y, 50):.0f}'
    )


@torch.no_grad()
def evaluate(head: PairwiseDistanceHead, z_i, z_j, y, label_scale, huber_beta, batch_size=8192):
    """Ranking metrics first, regression second.

    The head is a RANKER: one that regresses to the segment mean scores
    acceptably on RMSE and is useless for ordering, which is why Spearman is the
    headline and RMSE is a diagnostic column.
    """
    head.eval()
    preds_t = torch.cat([
        head(z_i[k : k + batch_size], z_j[k : k + batch_size])
        for k in range(0, z_i.shape[0], batch_size)
    ]).float()
    y_norm = y.float() / label_scale
    loss = F.smooth_l1_loss(preds_t, y_norm, beta=huber_beta).item()

    preds = preds_t.cpu().numpy() * label_scale
    truth = y.float().cpu().numpy()

    rho = float(spearmanr(preds, truth).statistic)
    rmse = float(np.sqrt(np.mean((preds - truth) ** 2)))

    # Fraction of top-decile-delta pairs ranked above bottom-decile ones, i.e.
    # the Mann-Whitney statistic between the two groups.
    hi = preds[truth >= np.quantile(truth, 0.9)]
    lo = preds[truth <= np.quantile(truth, 0.1)]
    ranks = rankdata(np.concatenate([hi, lo]))
    auc = float((ranks[: len(hi)].sum() - len(hi) * (len(hi) + 1) / 2) / (len(hi) * len(lo)))

    # Asymmetry gap, relative to the mean prediction: large means the
    # random-order augmentation never took (a training red flag, not a property
    # of the deployment query).
    gap = head.asymmetry(z_i[:8192], z_j[:8192]).mean().item() * label_scale
    return {
        'val/loss': loss,
        'val/spearman': rho,
        'val/rmse_frames': rmse,
        'val/decile_auc': auc,
        'val/asymmetry_ratio': gap / max(float(np.mean(preds)), 1e-6),
    }


@hydra.main(version_base=None, config_path='../../configs/trm', config_name='train')
def run(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(cfg.out_root or LW_ROOT / 'data' / 'trm' / 'heads') / cfg.output_model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # --- CACHE -------------------------------------------------------------
    path = cfg.latent_cache or default_cache_path(cfg.dataset, cfg.checkpoint, cfg.get('cache_dir'))
    cache = load_cache(path, expect_checkpoint=cfg.checkpoint, expect_dataset=cfg.dataset)
    logging.info(f'latents {cache.emb.shape} over {cache.n_episodes} episodes from {path}')
    if cache.dim != cfg.head.dim:
        raise ValueError(f'cache dim {cache.dim} != head dim {cfg.head.dim}')

    # --- PAIRS -------------------------------------------------------------
    # ONE seeded rng, drawn once up front: the pair table IS the budget, and the
    # shuffled control has to train on byte-identical pairs.
    rng = np.random.default_rng(cfg.seed)
    lengths, offsets = build_segments(
        cache.episode_idx,
        goal_met=cache.columns.get('goal_met'),
        split_on_goal_reset=cfg.sampler.split_on_goal_reset,
    )
    train_seg, val_seg = split_segments(len(lengths), cfg.sampler.val_frac, rng)
    logging.info(f'{len(lengths)} segments -> {len(train_seg)} train / {len(val_seg)} val')
    np.save(run_dir / 'val_segments.npy', val_seg)

    i_tr, j_tr, y_tr = draw_pairs(
        cache=cache,
        segments=(lengths[train_seg], offsets[train_seg]),
        cfg=cfg,
        n_pairs=cfg.sampler.n_pairs,
        rng=rng,
        flip=cfg.sampler.flip,
    )
    label_stats('train', y_tr)

    # Validation pairs are always DIRECTED (flip=False): the ordering ablation
    # compares held-out Spearman on directed pairs, so the val set must not
    # itself be order-randomized or the two arms are not comparable.
    i_va, j_va, y_va = draw_pairs(
        cache=cache,
        segments=(lengths[val_seg], offsets[val_seg]),
        cfg=cfg,
        n_pairs=cfg.sampler.n_val_pairs,
        rng=rng,
        flip=False,
    )
    label_stats('val', y_va)

    if cfg.shuffle_labels:
        logging.warning('NEGATIVE CONTROL: training labels are permuted')
        y_tr = shuffle_labels(y_tr, rng)

    # li26 uses 224. Long episodes put targets past Smooth-L1's beta=1 elbow,
    # quietly down-weighting the long-horizon pairs the method exists to get
    # right. Default to the observed max instead, so targets land in [0, 1].
    label_scale = float(cfg.label_scale or y_tr.max())
    logging.info(f'label_scale={label_scale:.1f} [(Li, 2026) uses scale=224]')

    z_i_tr, z_j_tr, y_t = pairs_to_device(cache.emb, i_tr, j_tr, y_tr, device)
    z_i_va, z_j_va, y_v = pairs_to_device(cache.emb, i_va, j_va, y_va, device)

    # --- head --------------------------------------------------------------
    head = hydra.utils.instantiate(cfg.head).to(device)
    n_pairs, batch = z_i_tr.shape[0], cfg.trainer.batch_size
    steps_per_epoch = max(n_pairs // batch, 1)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps_per_epoch * cfg.trainer.max_epochs
    )

    history = []
    metrics = {}
    for epoch in range(cfg.trainer.max_epochs):
        head.train()
        perm = torch.randperm(n_pairs, device=device)
        batches = perm.split(batch)
        running = 0.0
        for b in batches:
            pred = head(z_i_tr[b], z_j_tr[b])
            loss = F.smooth_l1_loss(
                pred,
                y_t[b] / label_scale,
                beta=cfg.huber_beta
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        metrics = evaluate(head, z_i_va, z_j_va, y_v, label_scale, cfg.huber_beta)
        metrics['train/loss'] = running / max(len(batches), 1)
        logging.info(
            f'epoch {epoch + 1}: ' + '  '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        )
        history.append(dict(epoch=epoch + 1, **metrics))

        if (epoch + 1) % cfg.trainer.save_epoch_interval == 0 or epoch + 1 == cfg.trainer.max_epochs:
            # config=cfg.head ONLY: load_head builds the head from it
            save_head(
                head,
                run_dir,
                config=OmegaConf.to_container(cfg.head, resolve=True),
                filename=f'weights_epoch_{epoch + 1}.pt',
            )
            (run_dir / 'trm_label_scale.json').write_text(
                json.dumps({'label_scale': label_scale, 'label': cfg.label}, indent=2)
            )

    (run_dir / 'metrics.json').write_text(json.dumps(history, indent=2))
    logging.info(f'done ({run_dir}). final: ' + '  '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
    if metrics.get('val/spearman', 0.0) < cfg.spearman_floor:
        logging.warning(
            f"held-out Spearman {metrics.get('val/spearman', 0.0):.3f} < {cfg.spearman_floor} "
            '(expected near 0 only when shuffle_labels=true, which is the control working correctly)'
        )


if __name__ == '__main__':
    run()
