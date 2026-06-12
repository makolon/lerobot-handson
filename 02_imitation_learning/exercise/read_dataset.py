#!/usr/bin/env python
# =============================================================================
# 02_imitation_learning/exercise/read_dataset.py
#   ANSWER to the Section 2 hands-on (Q5): read a real LeRobotDataset off the
#   object — fps, number of episodes, and every feature's dtype/shape — then
#   point out which keys are camera images and what the action dimension is.
#
# Runs on the login node, inside the container — no GPU, no job needed.
#
# Usage:
#   source config.env
#   apptainer exec $APPTAINER_IMAGE python 02_imitation_learning/exercise/read_dataset.py
#
#   # or inspect a specific repo / local root instead of $DATA_REPO:
#   apptainer exec $APPTAINER_IMAGE python 02_imitation_learning/exercise/read_dataset.py \
#       --repo-id HuggingFaceVLA/libero --root "$LIBERO_ROOT"
#
# Reads the pre-downloaded cache in $HF_HOME (set HF_HUB_OFFLINE=1 on the compute
# node); on the login node it can also download on demand.
# =============================================================================
import argparse
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--repo-id", default=os.environ.get("DATA_REPO"),
                help="dataset repo_id (default: $DATA_REPO)")
ap.add_argument("--root", default=os.environ.get("LEROBOT_DATASET_ROOT"),
                help="optional local dataset dir (e.g. $LIBERO_ROOT) -> reads offline")
args = ap.parse_args()

if not args.repo_id:
    raise SystemExit("set --repo-id or $DATA_REPO (source config.env first)")

# --- 1. load + the headline numbers ----------------------------------------
ds = LeRobotDataset(args.repo_id, root=args.root) if args.root \
    else LeRobotDataset(args.repo_id)
print(f"repo_id : {args.repo_id}")
print(f"fps     : {ds.meta.fps}")
print(f"episodes: {ds.num_episodes}")
print(f"frames  : {ds.num_frames}")

# --- 2. the schema: every key, its dtype and shape -------------------------
print("\n=== features ===")
image_keys, action_dim = [], None
for name, spec in ds.meta.features.items():
    shape = tuple(spec["shape"])
    print(f"  {name:28s} {spec['dtype']:8s} {shape}")
    if name.startswith("observation.images."):
        image_keys.append(name)
    if name == "action":
        action_dim = shape

# --- 3. read it off the object (the answers to Q5) -------------------------
print("\n=== read it off the object ===")
print(f"  action dimension : {action_dim}")
print(f"  camera image keys: {image_keys or '(none)'}")
print("  -> observation.* and action are the DATA; the same observation/action")
print("     format is the contract the ENVIRONMENT must match at eval time.")

# one decoded frame: note images come back channels-first (C, H, W) for torch
frame = ds[0]
print("\n=== sample frame[0] shapes (note: images are channels-first) ===")
for k in list(image_keys) + (["observation.state", "action"]):
    if k in frame and hasattr(frame[k], "shape"):
        print(f"  {k:28s} {tuple(frame[k].shape)}")
print(f"  task                         {frame.get('task')!r}")
