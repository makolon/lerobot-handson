#!/usr/bin/env bash
# =============================================================================
# download_libero.sh  —  fetch the LeRobot LIBERO dataset into the SHARED dir
# -----------------------------------------------------------------------------
# Run on the Miyabi [LOGIN NODE] (it has internet; compute nodes are offline).
#
# What it does (everything lands in the shared area so the whole class reuses it):
#   1. downloads HuggingFaceVLA/libero (a LeRobotDataset combining the LIBERO
#      suites: spatial / object / goal / 10) into  $LIBERO_ROOT
#   2. warm-loads it once with lerobot so any dataset-version conversion happens
#      now, online — not later on the offline compute node
#   3. pre-caches the ACT ResNet18 ImageNet backbone into $TORCH_HOME, so offline
#      training doesn't try to download it on the compute node
#
# After this, train offline from the share with 04_policy_training/train_libero.{sh,pbs}.
#
# Override any of these before running (defaults target /work/gw13/share/handson):
#   SHARE_DIR     shared root            (default /work/gw13/share/handson)
#   LIBERO_REPO   HF dataset repo_id     (default HuggingFaceVLA/libero)
#   LIBERO_ROOT   where the dataset lands(default $SHARE_DIR/libero)  <- train --dataset.root
#   TORCH_HOME    backbone cache         (default $SHARE_DIR/torch)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

# config.env gives us APPTAINER_IMAGE / APPTAINER_MODULE / HF_HOME (and lets the
# user override the paths below). It is optional but recommended.
if [[ -f "${REPO}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO}/config.env"
fi

: "${APPTAINER_IMAGE:?APPTAINER_IMAGE unset — source config.env and build the image (01_hpc) first}"
: "${SHARE_DIR:=/work/gw13/share/handson}"
: "${LIBERO_REPO:=HuggingFaceVLA/libero}"
: "${LIBERO_ROOT:=${SHARE_DIR}/libero}"
: "${TORCH_HOME:=${SHARE_DIR}/torch}"
: "${HF_HOME:=${SHARE_DIR}/hf_home}"   # download scratch/cache (kept off the small $HOME)

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: image not found: ${APPTAINER_IMAGE}  (build it: bash env/build_image.sh)" >&2
  exit 1
fi

mkdir -p "${LIBERO_ROOT}" "${TORCH_HOME}" "${HF_HOME}"

# Apptainer module (so `apptainer` is on PATH on the login node).
if [[ -n "${APPTAINER_MODULE:-}" && "${APPTAINER_MODULE}" != "<"* ]]; then
  module load "${APPTAINER_MODULE}"
fi

echo "[libero] repo       : ${LIBERO_REPO}"
echo "[libero] dataset -> : ${LIBERO_ROOT}"
echo "[libero] backbone ->: ${TORCH_HOME}"
echo "[libero] image      : ${APPTAINER_IMAGE}"

# We run hf / python from INSIDE the container so the right
# huggingface_hub + lerobot + torchvision are guaranteed present (no reliance on
# the login node's Python). Apptainer has network on the login node.
#
# CRITICAL: --home forces $HOME onto /work. By default Apptainer keeps $HOME at
# /home/$USER, and HF/xet write transient caches under ~/.cache. /home is a small
# Lustre fs (≈50 GB, often nearly full), so a 15 GB download fills it and dies with
# "Disk quota exceeded (os error 122)". With $HOME and HF_XET_CACHE on the roomy
# /work, every cache stays there and the download completes.
# HF_HUB_DISABLE_XET=1: use plain HTTPS, NOT the xet protocol. xet keeps a growing
# chunk cache during a large multi-file download that overruns a quota partway
# through (it died ~file 131/383); plain HTTPS streams each file straight to the
# local dir with no cumulative cache. Slightly slower, but reliable for a one-off.
APPTAINER_RUN=(apptainer exec
  --home "${HF_HOME}"
  --bind "${LIBERO_ROOT}:${LIBERO_ROOT}"
  --bind "${TORCH_HOME}:${TORCH_HOME}"
  --env "HF_HOME=${HF_HOME}"
  --env "HF_HUB_DISABLE_XET=1"
  --env "TORCH_HOME=${TORCH_HOME}"
  "${APPTAINER_IMAGE}")

# --- 1. download the dataset files into the shared dir ------------------------
echo "== [1/3] downloading dataset =="
"${APPTAINER_RUN[@]}" \
  hf download "${LIBERO_REPO}" --repo-type dataset --local-dir "${LIBERO_ROOT}"

# --- 2. warm-load once (online) so any v2.x -> v3 conversion is cached here ---
echo "== [2/3] warm-loading the dataset (caches any format conversion) =="
if ! "${APPTAINER_RUN[@]}" python - "${LIBERO_REPO}" "${LIBERO_ROOT}" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id, root = sys.argv[1], sys.argv[2]
ds = LeRobotDataset(repo_id, root=root)
print(f"[libero] loaded OK: {ds.num_episodes} episodes, {ds.num_frames} frames")
PY
then
  echo "WARN: warm-load failed. The files are downloaded, but if offline training" >&2
  echo "      hits a dataset-version error, re-run this step online." >&2
fi

# --- 3. pre-cache the ACT ResNet18 ImageNet backbone -------------------------
# ACT's vision backbone defaults to torchvision ResNet18 IMAGENET1K_V1, fetched
# via torch.hub into TORCH_HOME. Cache it now so the offline compute node finds it.
# (Alternatively train with PRETRAINED_BACKBONE_WEIGHTS=null for random init.)
echo "== [3/3] pre-caching ACT ResNet18 ImageNet backbone =="
"${APPTAINER_RUN[@]}" python - <<'PY'
import torchvision
torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
print("[libero] ResNet18 ImageNet weights cached under TORCH_HOME")
PY

echo
echo "[libero] done."
echo "  dataset : ${LIBERO_ROOT}    (du -sh to check size)"
echo "  backbone: ${TORCH_HOME}"
echo "  train it: source config.env && qsub -q \"\$QUEUE_NAME\" -W group_list=\"\$GROUP\" \\"
echo "              -l select=1 -l walltime=\"\$WALLTIME\" 04_policy_training/train_libero.pbs"
