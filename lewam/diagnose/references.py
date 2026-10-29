"""Latent banks of dataset frames and kNN distances."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from lewam.eval.episodes import img_transform


@dataclass
class LatentBank:
    """Encoded dataset frames under one checkpoint."""

    latents: torch.Tensor        # (N, z)
    episodes: torch.Tensor       # (N,) dataset episode index of each row
    steps: torch.Tensor          # (N,) step within the episode
    dataset: str
    checkpoint: str
    half: int | None = None      # which episode half of the dataset, or None for all episodes

    def knn(self, z: torch.Tensor, exclude_episode: int | None = None, k: int = 5) -> torch.Tensor:
        """(M,) mean per-dim MSE from each row of z (M, z) to its k nearest bank latents."""
        latents = self.latents.to(z.device)
        dist = torch.cdist(z, latents) ** 2 / z.shape[-1]
        if exclude_episode is not None:
            dist[:, self.episodes.to(z.device) == exclude_episode] = float("inf")
        return dist.topk(k, dim=1, largest=False).values.mean(1)

    def save(self, path: str | Path) -> None:
        torch.save(dict(self.__dict__), path)

    @classmethod
    def load(cls, path: str | Path) -> "LatentBank":
        return cls(**torch.load(path))


def sample_bank_episodes(ep_len: np.ndarray, n_frames: int, seed: int, half: int | None = None) -> list[int]:
    """Episodes in a seeded random order until their frames reach n_frames; half 0 / 1 draws from disjoint halves."""
    order = np.random.default_rng(seed).permutation(len(ep_len))
    if half is not None:
        order = np.array_split(order, 2)[half]
    chosen, total = [], 0
    for episode in order:
        if total >= n_frames:
            break
        chosen.append(int(episode))
        total += int(ep_len[episode])
    assert total >= n_frames, f"only {total} frames available, {n_frames} requested"
    return chosen


@torch.no_grad()
def build_bank(model, h5_path: str | Path, checkpoint: str, n_frames: int, seed: int = 0,
               half: int | None = None, img_size: int = 224, batch_size: int = 256) -> LatentBank:
    """Encode n_frames dataset frames (whole episodes, the last one trimmed) with the eval's pixel transform."""
    device = next(model.parameters()).device
    transform = img_transform(img_size)
    with h5py.File(h5_path, "r") as data:
        ep_offset, ep_len = data["ep_offset"][:], data["ep_len"][:]
        latents, row_episodes, row_steps = [], [], []
        for episode in sample_bank_episodes(ep_len, n_frames, seed, half):
            n_rows = min(int(ep_len[episode]), n_frames - len(row_steps))
            start = int(ep_offset[episode])
            pixels = data["pixels"][start:start + n_rows]
            for batch_start in range(0, n_rows, batch_size):
                frames = pixels[batch_start:batch_start + batch_size]
                batch = torch.stack([transform(frame) for frame in frames]).to(device)
                latents.append(model.encode(batch).float().cpu())
            row_episodes += [episode] * n_rows
            row_steps += list(range(n_rows))
    return LatentBank(
        latents=torch.cat(latents),
        episodes=torch.tensor(row_episodes),
        steps=torch.tensor(row_steps),
        dataset=Path(h5_path).stem,
        checkpoint=checkpoint,
        half=half,
    )
