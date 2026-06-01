#!/usr/bin/env python
# =============================================================================
# convert_sample.py  —  minimal example: convert synthetic data to LeRobotDataset (v3.0)
# -----------------------------------------------------------------------------
# Goal: define features/fps/robot_type and experience the sequence
#       add_frame -> save_episode -> finalize. Use as a template for real-data conversion.
#
# Usage:
#   python 02_convert/convert_sample.py          # local save only
#   python 02_convert/convert_sample.py --push   # push to a repo under HF_USER
#
# TODO(lerobot): confirm the exact signatures of LeRobotDataset.create / add_frame /
#                save_episode / finalize against the v0.5.1 docs / --help.
#                (The fact that finalize() became mandatory in v3.0 is confirmed: PR #1903)
# =============================================================================
import argparse
import os

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_features(state_dim: int, action_dim: int, image_hw=(96, 96)):
    """Define dtype / shape / names for each key.

    - observation.state, action : low-dimensional continuous vectors (float32)
    - observation.images.front  : camera frames (H, W, C) uint8 -> video-encoded internally
    """
    h, w = image_hw
    return {
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
            "dtype": "video",  # image sequences are video-encoded
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="push to a repo under HF_USER")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    args = parser.parse_args()

    # --- decide repo_id (HF_USER required when pushing: fail-fast) ---
    hf_user = os.environ.get("HF_USER", "")
    if args.push:
        if not hf_user or hf_user.startswith("<TODO"):
            raise SystemExit("ERROR: --push requires HF_USER. Run `source config.env`.")
    repo_id = f"{hf_user or 'local-user'}/handson-convert-sample"

    image_hw = (96, 96)
    features = build_features(args.state_dim, args.action_dim, image_hw)

    # --- create a new dataset ---
    # TODO(lerobot): confirm create() arg names (fps/features/robot_type/use_videos, ...).
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        robot_type="handson_dummy_arm",
        use_videos=True,
    )

    rng = np.random.default_rng(0)
    total_frames = 0
    for ep in range(args.episodes):
        for _ in range(args.frames_per_episode):
            frame = {
                "observation.state": torch.from_numpy(
                    rng.standard_normal(args.state_dim).astype(np.float32)
                ),
                "action": torch.from_numpy(
                    rng.standard_normal(args.action_dim).astype(np.float32)
                ),
                "observation.images.front": (
                    rng.integers(0, 256, size=(*image_hw, 3), dtype=np.uint8)
                ),
            }
            # TODO(lerobot): confirm the arg name for passing a task string to add_frame
            #                (v0.5.1 attaches a task per frame/episode).
            dataset.add_frame(frame, task="pick up the cube")
            total_frames += 1
        dataset.save_episode()

    # --- always finalize (parquet is corrupt if you don't) ---
    dataset.finalize()
    print(f"[convert] wrote {args.episodes} episodes / {total_frames} frames -> {repo_id}")

    if args.push:
        # Needs network; run on the login node
        dataset.push_to_hub()
        print(f"[convert] pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
