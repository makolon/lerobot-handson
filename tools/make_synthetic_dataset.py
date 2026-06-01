#!/usr/bin/env python
# =============================================================================
# make_synthetic_dataset.py  —  generate synthetic robot data, no net / no GPU
# -----------------------------------------------------------------------------
# Purpose: make the 01_dataset / 02_convert / 03_train exercises runnable without
#          the HF Hub or Miyabi. Produces small dummy episodes of
#          observation.state / action / observation.images.front (+ a task string).
#
# Two output formats:
#   --format lerobot  (default) : write a ready-to-load LeRobotDataset (v3.0) under --root.
#                                 Used by 01_dataset (inspect) and 03_train (train on it).
#   --format raw                : write raw per-episode .npz files under --out.
#                                 Consumed by 02_convert/convert_sample.py to demonstrate
#                                 the create -> add_frame -> save_episode -> finalize flow.
#
# Runs on CPU with no network. Verified with lerobot==0.5.1.
# =============================================================================
import argparse
import shutil
from pathlib import Path

import numpy as np


def gen_episode(rng, length, state_dim, action_dim, hw):
    """Return (states, actions, images) for one synthetic episode.

    The signals are smooth random walks so plots in 01_dataset look like a
    trajectory rather than pure noise. Images are small random tiles.
    """
    h, w = hw
    states = np.cumsum(rng.standard_normal((length, state_dim)).astype(np.float32) * 0.1, axis=0)
    # action ~ next state delta (a toy but plausible relationship); repeat last row to keep length
    actions = np.diff(states, axis=0, append=states[-1:]).astype(np.float32)
    images = rng.integers(0, 256, size=(length, h, w, 3), dtype=np.uint8)
    return states, actions, images


def write_lerobot(args, rng):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)

    h, w = args.height, args.width
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (args.state_dim,),
            "names": [f"state_{i}" for i in range(args.state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (args.action_dim,),
            "names": [f"action_{i}" for i in range(args.action_dim)],
        },
        # "image" (not "video") keeps the smoke test free of video codecs.
        "observation.images.front": {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type="handson_dummy_arm",
        use_videos=False,
    )
    total = 0
    for _ in range(args.episodes):
        states, actions, images = gen_episode(
            rng, args.length, args.state_dim, args.action_dim, (h, w)
        )
        for t in range(args.length):
            ds.add_frame(
                {
                    "observation.state": states[t],
                    "action": actions[t],
                    "observation.images.front": images[t],
                    "task": args.task,
                }
            )
            total += 1
        ds.save_episode()
    ds.finalize()
    print(f"[make] LeRobotDataset written: {root}")
    print(f"[make]   episodes={ds.num_episodes} frames={total} fps={args.fps} "
          f"state_dim={args.state_dim} action_dim={args.action_dim} image={h}x{w}")


def write_raw(args, rng):
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    h, w = args.height, args.width
    for ep in range(args.episodes):
        states, actions, images = gen_episode(
            rng, args.length, args.state_dim, args.action_dim, (h, w)
        )
        np.savez(
            out / f"episode_{ep:03d}.npz",
            observation_state=states,
            action=actions,
            observation_image_front=images,
            task=np.array(args.task),
        )
    meta = {"fps": args.fps, "state_dim": args.state_dim, "action_dim": args.action_dim,
            "height": h, "width": w, "episodes": args.episodes}
    (out / "meta.txt").write_text("\n".join(f"{k}={v}" for k, v in meta.items()))
    print(f"[make] raw episodes written: {out} ({args.episodes} .npz files)")


def main():
    p = argparse.ArgumentParser(description="Generate synthetic robot data (no net/GPU).")
    p.add_argument("--format", choices=["lerobot", "raw"], default="lerobot")
    p.add_argument("--root", default=".smoke/synthetic", help="LeRobotDataset output dir (format=lerobot)")
    p.add_argument("--out", default=".smoke/raw", help="raw .npz output dir (format=raw)")
    p.add_argument("--repo-id", default="handson/synthetic")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--length", type=int, default=24, help="frames per episode")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--state-dim", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--task", default="pick up the cube")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.format == "lerobot":
        write_lerobot(args, rng)
    else:
        write_raw(args, rng)


if __name__ == "__main__":
    main()
