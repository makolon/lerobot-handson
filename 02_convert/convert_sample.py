#!/usr/bin/env python
# =============================================================================
# convert_sample.py  —  convert raw episodes into a LeRobotDataset (v3.0)
# -----------------------------------------------------------------------------
# Reads the raw .npz episodes produced by tools/make_synthetic_dataset.py
# (--format raw) and builds a LeRobotDataset by defining features / fps /
# robot_type and calling create -> add_frame -> save_episode -> finalize.
# After writing, it reloads the dataset and asserts the shapes round-trip.
#
# This is the template for converting *real* robot data: keep the structure,
# swap the .npz reader for your own raw source.
#
# Usage (offline, no GPU):
#   python tools/make_synthetic_dataset.py --format raw --out .smoke/raw
#   python 02_convert/convert_sample.py --raw .smoke/raw --root .smoke/converted
#   python 02_convert/convert_sample.py --raw .smoke/raw --root .smoke/converted --push  # to the Hub
#
# Verified with lerobot==0.5.1.
# =============================================================================
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def read_meta(raw_dir: Path) -> dict:
    meta = {}
    for line in (raw_dir / "meta.txt").read_text().splitlines():
        k, v = line.split("=", 1)
        meta[k] = int(v)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=".smoke/raw", help="dir of raw .npz episodes")
    p.add_argument("--root", default=".smoke/converted", help="output LeRobotDataset dir")
    p.add_argument("--repo-id", default=None, help="defaults to ${HF_USER}/handson-convert-sample")
    p.add_argument("--push", action="store_true", help="push to the Hub (needs HF_USER, network)")
    args = p.parse_args()

    raw_dir = Path(args.raw)
    episode_files = sorted(raw_dir.glob("episode_*.npz"))
    if not episode_files:
        raise SystemExit(f"ERROR: no episode_*.npz in {raw_dir}. Run tools/make_synthetic_dataset.py --format raw first.")
    meta = read_meta(raw_dir)

    # --- repo_id (HF_USER required only when pushing: fail-fast) ---
    hf_user = os.environ.get("HF_USER", "")
    if args.push and (not hf_user or hf_user.startswith("<TODO")):
        raise SystemExit("ERROR: --push requires HF_USER. Run `source config.env`.")
    repo_id = args.repo_id or f"{hf_user or 'local-user'}/handson-convert-sample"

    state_dim, action_dim = meta["state_dim"], meta["action_dim"]
    h, w = meta["height"], meta["width"]

    # --- define the schema: dtype / shape / names per key ---
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    }

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=meta["fps"],
        features=features,
        root=root,
        robot_type="handson_dummy_arm",
        use_videos=False,
    )

    total = 0
    for ep_file in episode_files:
        z = np.load(ep_file)
        states = z["observation_state"]
        actions = z["action"]
        images = z["observation_image_front"]
        task = str(z["task"])
        for t in range(len(states)):
            ds.add_frame(
                {
                    "observation.state": states[t].astype(np.float32),
                    "action": actions[t].astype(np.float32),
                    "observation.images.front": images[t],
                    "task": task,  # the 'task' key is required by add_frame
                }
            )
            total += 1
        ds.save_episode()

    # Must finalize, or the parquet files stay incomplete and won't load.
    ds.finalize()
    print(f"[convert] wrote {len(episode_files)} episodes / {total} frames -> {root} (repo_id={repo_id})")

    # --- verify the round-trip ---
    reloaded = LeRobotDataset(repo_id, root=root)
    sample = reloaded[0]
    assert tuple(sample["action"].shape) == (action_dim,), sample["action"].shape
    assert tuple(sample["observation.state"].shape) == (state_dim,), sample["observation.state"].shape
    assert tuple(sample["observation.images.front"].shape) == (3, h, w), sample["observation.images.front"].shape
    assert reloaded.num_episodes == len(episode_files)
    print(f"[convert] reload OK: episodes={reloaded.num_episodes} frames={reloaded.num_frames} "
          f"action{tuple(sample['action'].shape)} image{tuple(sample['observation.images.front'].shape)}")

    if args.push:
        reloaded.push_to_hub()  # network; run on the login node
        print(f"[convert] pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
