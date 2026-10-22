"""TRM head checkpoints: `<run_dir>/weights_epoch_<N>.pt` (state_dict) + `config.json` (head config).

Same layout as stable-worldmodel's `save_pretrained`, so heads convert between the two repos by copying files.
"""

import json
from pathlib import Path

import torch

from lewam.trm.head import PairwiseDistanceHead


def save_head(head: PairwiseDistanceHead, run_dir: Path, config: dict, filename: str) -> Path:
    """Write the head's state_dict and its constructor config into run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    torch.save(head.state_dict(), path)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return path


def load_head(weights_path: str | Path) -> PairwiseDistanceHead:
    """A head from a weights file and the config.json next to it."""
    weights_path = Path(weights_path)
    config = json.loads((weights_path.parent / "config.json").read_text())
    config.pop("_target_", None)
    head = PairwiseDistanceHead(**config)
    head.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return head
