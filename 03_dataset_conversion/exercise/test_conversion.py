#!/usr/bin/env python
# =============================================================================
# 03_dataset_conversion/test_conversion.py
#   Reload the LeRobotDataset you produced and check it is correct. Prints OK if
#   everything passes; otherwise an assertion tells you exactly what is wrong.
#
# Usage (login node, inside the container — no GPU needed):
#   apptainer exec $APPTAINER_IMAGE python 03_dataset_conversion/test_conversion.py \
#       --root .out/libero
#
#   # to also check the episode count against the source file:
#   apptainer exec $APPTAINER_IMAGE python 03_dataset_conversion/test_conversion.py \
#       --root .out/libero --hdf5 "$LIBERO_TASK_HDF5"
# =============================================================================
import argparse
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REQUIRED_FEATURES = [
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "action",
]
EXPECTED_FPS = 20          # LIBERO / robosuite control runs at 20 Hz
EXPECTED_STATE_DIM = 8     # ee_pos(3) + ee_ori(3) + gripper(2)
EXPECTED_ACTION_DIM = 7    # 6-DoF delta pose + gripper

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="root dir you converted into")
ap.add_argument("--repo-id", default="local-user/libero-task",
                help="must match the repo_id used in convert_libero_hdf5.py")
ap.add_argument("--hdf5", default=os.environ.get("LIBERO_TASK_HDF5"),
                help="source LIBERO hdf5 (default: $LIBERO_TASK_HDF5); "
                     "if given, checks the episode count")
args = ap.parse_args()

# --- 1. it loads -----------------------------------------------------------
assert os.path.isdir(args.root), f"root not found: {args.root}"
ds = LeRobotDataset(args.repo_id, root=args.root)
print(f"[test] loaded: {ds.num_episodes} episodes, {len(ds)} frames, fps={ds.meta.fps}")

# --- 2. episode count matches the source hdf5 (optional) -------------------
if args.hdf5 and os.path.exists(args.hdf5):
    import h5py
    f = h5py.File(args.hdf5, "r")
    n_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
    f.close()
    assert ds.num_episodes == n_demos, \
        f"episode count {ds.num_episodes} != {n_demos} demos in the hdf5"
    print(f"[test] episode count matches the source hdf5 ({n_demos})")
else:
    print("[test] (no --hdf5 / $LIBERO_TASK_HDF5 -> skipping episode-count check)")

# --- 3. the required features exist ----------------------------------------
feats = ds.meta.features
for k in REQUIRED_FEATURES:
    assert k in feats, f"missing feature: {k}"
print("[test] all required features present:", ", ".join(REQUIRED_FEATURES))

# --- 4. fps is correct -----------------------------------------------------
assert ds.meta.fps == EXPECTED_FPS, \
    f"fps is {ds.meta.fps}, expected {EXPECTED_FPS} for LIBERO"

# image H, W come from the feature spec (shape is H, W, C)
H, W, C = feats["observation.images.image"]["shape"]
assert C == 3, f"image channel dim is {C}, expected 3"

# --- 5. sampled frames have the right shapes and a real task ---------------
n = len(ds)
assert n > 0, "dataset has no frames"
idxs = sorted(set([0, n - 1] + [int(i * (n - 1) / 9) for i in range(10)]))

for j, i in enumerate(idxs):
    frame = ds[i]

    if j == 0:  # shape checks once is enough (every frame shares the schema)
        assert tuple(frame["observation.state"].shape) == (EXPECTED_STATE_DIM,), \
            f"state shape {tuple(frame['observation.state'].shape)} != ({EXPECTED_STATE_DIM},)"
        assert tuple(frame["action"].shape) == (EXPECTED_ACTION_DIM,), \
            f"action shape {tuple(frame['action'].shape)} != ({EXPECTED_ACTION_DIM},)"
        for key in ("observation.images.image", "observation.images.image2"):
            assert tuple(frame[key].shape) == (3, H, W), \
                f"{key} shape {tuple(frame[key].shape)} != (3, {H}, {W}) " \
                f"(images must be channels-first)"

    task = frame["task"]
    assert isinstance(task, str) and len(task.strip()) > 0, \
        f"frame {i}: task is empty (did you pass 'task' to every add_frame?)"

print(f"[test] shapes OK: state ({EXPECTED_STATE_DIM},), action ({EXPECTED_ACTION_DIM},), "
      f"images (3, {H}, {W})")
print(f"[test] task string present on all {len(idxs)} sampled frames "
      f"(e.g. \"{ds[0]['task']}\")")

print("OK")