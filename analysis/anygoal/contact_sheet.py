"""Contact sheets of failed AnyGoal episodes: the goal frame, then every agent frame with its index.

Reads env_<i>.mp4 (panels agent | dataset | goal) and results.txt from
data/beam_smoke/<env>/anygoal_video/<arm>/<seed>/, writes data/beam_smoke/analysis/anygoal_video/<env>/. Run from
playground/lewam:
    /home/eben/.conda/envs/lewam/bin/python analysis/anygoal/contact_sheet.py avoidt stackt
"""

import re
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

LW = Path(__file__).resolve().parents[2]
SMOKE = LW / "data" / "beam_smoke"
OUT = SMOKE / "analysis" / "anygoal_video"
ARM = "native_grad"
SEEDS = [8902, 10748, 17401, 18714, 28967, 30531, 36136, 37182, 47259, 60813]
MIN_FAILURES = 5
COLS = 10
PANEL = 224
TOP = 16
AGENT_X, GOAL_X = 16, 492   # left edges of the agent and goal panels in the 736 x 288 video frame


def failures(run_dir):
    """Episode slots marked False in results.txt."""
    text = (run_dir / "results.txt").read_text()
    flags = re.search(r"episode_successes': array\(\[([^\]]*)\]", text).group(1)
    return [i for i, flag in enumerate(flags.replace(",", " ").split()) if flag == "False"]


def tile(frame, label):
    """Frame with its label in the top-left corner."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 + 7 * len(label), 14], fill=(255, 255, 255))
    draw.text((3, 2), label, fill=(0, 0, 0))
    return np.asarray(img)


def contact_sheet(video_path):
    """Goal tile, then all agent tiles, COLS per row."""
    frames = iio.imread(video_path)
    rows_px = slice(TOP, TOP + PANEL)
    agent = frames[:, rows_px, AGENT_X:AGENT_X + PANEL]
    goal = frames[0, rows_px, GOAL_X:GOAL_X + PANEL]
    tiles = [tile(goal, "goal")] + [tile(f, str(t)) for t, f in enumerate(agent)]
    gap = np.full((PANEL, 4, 3), 255, dtype=np.uint8)
    tiles = [np.concatenate([t, gap], axis=1) for t in tiles]
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % COLS)
    rows = [np.concatenate(tiles[r:r + COLS], axis=1) for r in range(0, len(tiles), COLS)]
    return np.concatenate(rows, axis=0)


def main(envs):
    for env in envs:
        out_dir = OUT / env
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for seed in SEEDS:
            run_dir = SMOKE / env / "anygoal_video" / ARM / str(seed)
            if not (run_dir / "results.txt").exists():
                continue
            for slot in failures(run_dir):
                sheet = contact_sheet(run_dir / f"env_{slot}.mp4")
                iio.imwrite(out_dir / f"{seed}_env{slot}.png", sheet)
                written += 1
        print(f"{env}: {written} failed episodes written to {out_dir}")
        if written < MIN_FAILURES:
            print(f"{env}: fewer than {MIN_FAILURES} failures, run more seeds")


if __name__ == "__main__":
    main(sys.argv[1:])
