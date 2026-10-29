"""M2 sanity checks of the latent banks: half-vs-half baseline and expert-frame pull, raw and L2-normalised.

  python scripts/diagnose/check_latent_banks.py --banks data/beam_smoke/latent_banks --n-query 2000
"""

import argparse
from pathlib import Path

import torch

from lewam.diagnose.references import LatentBank

# (name, query bank, task bank, train bank): queries are expert frames of the query bank
PAIRS = [
    ("PushT half1 vs half0", "lewam-pusht__pusht_expert_train__half1", "lewam-pusht__pusht_expert_train__half1",
     "lewam-pusht__pusht_expert_train__half0"),
    ("StackT", "lewam-pusht__stackt_test", "lewam-pusht__stackt_test", "lewam-pusht__pusht_expert_train__half0"),
    ("Cube half1 vs half0", "lewam-cube__cube__half1", "lewam-cube__cube__half1", "lewam-cube__cube__half0"),
    ("CubeDiagonal", "lewam-cube__cube_diagonal_test", "lewam-cube__cube_diagonal_test", "lewam-cube__cube__half0"),
    ("CubeHop", "lewam-cube__cube_hop_test", "lewam-cube__cube_hop_test", "lewam-cube__cube__half0"),
]


def normalised(bank: LatentBank) -> LatentBank:
    """Copy of bank with L2-normalised latents."""
    latents = torch.nn.functional.normalize(bank.latents, dim=-1)
    return LatentBank(latents, bank.episodes, bank.steps, bank.dataset, bank.checkpoint, bank.half)


def own_episode_knn(bank: LatentBank, z: torch.Tensor, episodes: torch.Tensor, k: int = 5) -> torch.Tensor:
    """kNN distance of each query row to bank, excluding that row's own episode."""
    return torch.cat([bank.knn(z[i:i + 1], exclude_episode=int(episodes[i]), k=k) for i in range(len(z))])


def pull_stats(query: LatentBank, task: LatentBank, train: LatentBank, n_query: int, device: str) -> dict:
    rows = torch.randperm(len(query.latents), generator=torch.Generator().manual_seed(0))[:n_query]
    z = query.latents[rows].to(device)
    d_task = own_episode_knn(task, z, query.episodes[rows])
    d_train = train.knn(z)
    pull = (d_task - d_train).cpu()
    return dict(d_task=d_task.median().item(), d_train=d_train.median().item(), pull_median=pull.median().item(),
                pull_q10=pull.quantile(0.1).item(), pull_q90=pull.quantile(0.9).item(),
                share_positive=(pull > 0).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--n-query", type=int, default=2000)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    banks = {path.stem: LatentBank.load(path) for path in Path(args.banks).glob("*.pt")}

    print("pair | space | d_task median | d_train median | pull median [q10, q90] | share pull > 0")
    for name, query, task, train in PAIRS:
        for space, transform in (("raw", lambda bank: bank), ("l2", normalised)):
            stats = pull_stats(transform(banks[query]), transform(banks[task]), transform(banks[train]),
                               args.n_query, device)
            print(f"{name} | {space} | {stats['d_task']:.4g} | {stats['d_train']:.4g} | "
                  f"{stats['pull_median']:+.4g} [{stats['pull_q10']:+.4g}, {stats['pull_q90']:+.4g}] | "
                  f"{stats['share_positive']:.2f}", flush=True)


if __name__ == "__main__":
    main()
