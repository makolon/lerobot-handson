#!/usr/bin/env python
# =============================================================================
# 03_dataset_conversion/download_libero_hdf5.py
#   Download ONE raw LIBERO task .hdf5 (robomimic/robosuite format) into the
#   shared area. Run on the Miyabi [login node] — it has internet. Compute nodes
#   are offline, so we stage the file once here and everyone reads it offline.
#
#   Source: yifengzhu-hf/LIBERO-datasets (the official raw LIBERO hdf5 demos).
#   The full repo is ~100 GB, so we fetch a SINGLE task file, not the whole thing.
#
# Usage (on the login node, inside the container):
#   apptainer exec $APPTAINER_IMAGE python 03_dataset_conversion/download_libero_hdf5.py \
#       --suite libero_object --dest "$SHARED_DIR/libero_raw"
#
#   # pick a specific task instead of the first one:
#   apptainer exec $APPTAINER_IMAGE python 03_dataset_conversion/download_libero_hdf5.py \
#       --suite libero_object --match alphabet_soup --dest "$SHARED_DIR/libero_raw"
#
# Then copy the printed path into config.env as LIBERO_TASK_HDF5.
# =============================================================================
import argparse
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "yifengzhu-hf/LIBERO-datasets"   # raw LIBERO hdf5, organized by suite

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_object",
                choices=["libero_object", "libero_spatial", "libero_goal",
                         "libero_10", "libero_90"],
                help="which LIBERO suite to pull a task from")
ap.add_argument("--dest", required=True,
                help="shared directory to stage the file into (e.g. $SHARED_DIR/libero_raw)")
ap.add_argument("--match", default=None,
                help="substring to select a specific task file (default: first task)")
args = ap.parse_args()

# list the task files in the chosen suite (each task = one *_demo.hdf5)
files = [f for f in list_repo_files(REPO, repo_type="dataset")
         if f.startswith(args.suite + "/") and f.endswith("_demo.hdf5")]
files.sort()
if args.match is not None:
    files = [f for f in files if args.match in f]
assert files, f"no matching task file in {args.suite} (match={args.match})"

target = files[0]
print(f"[download] repo={REPO}")
print(f"[download] suite={args.suite}  picked={target}")
if len(files) > 1:
    print(f"[download] ({len(files)} tasks matched; took the first. "
          f"use --match to choose another)")

local = hf_hub_download(
    repo_id=REPO,
    repo_type="dataset",
    filename=target,
    local_dir=args.dest,
)

print("[download] saved to:", local)
print()
print("Add this line to config.env (then re-source it):")
print(f'  export LIBERO_TASK_HDF5="{local}"')