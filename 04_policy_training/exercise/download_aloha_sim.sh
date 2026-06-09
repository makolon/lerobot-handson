#!/usr/bin/env bash
# =============================================================================
# download_aloha_sim.sh  —  ANSWER to Exercise Q1 (fetch dataset + SmolVLA base)
# -----------------------------------------------------------------------------
# Run on the Miyabi [LOGIN NODE] (it has internet; compute nodes are offline).
#
# Downloads, into the shared HF cache ($HF_HOME), the two things the SmolVLA
# fine-tune needs:
#   1. an ALOHA Sim dataset (default lerobot/aloha_sim_insertion_human)
#   2. the SmolVLA base model  lerobot/smolvla_base  (we fine-tune FROM it)
# then warm-loads the dataset once (online) so any version conversion is cached
# here rather than on the offline compute node.
#
# After this, train offline with exercise/train_smolvla.{sh,pbs}.
#
# Override before running:
#   SMOLVLA_DATASET   HF dataset repo_id   (default lerobot/aloha_sim_insertion_human)
#   SMOLVLA_BASE      base model repo_id   (default lerobot/smolvla_base)
#   HF_HOME           HF cache             (from config.env; the bind target)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

# config.env gives APPTAINER_IMAGE / APPTAINER_MODULE / HF_HOME.
if [[ -f "${REPO}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO}/config.env"
fi

: "${APPTAINER_IMAGE:?APPTAINER_IMAGE unset — source config.env and build the image (01_hpc) first}"
: "${SHARE_DIR:=/work/gw13/share/handson}"
: "${HF_HOME:=${SHARE_DIR}/hf_home}"
: "${SMOLVLA_DATASET:=lerobot/aloha_sim_insertion_human}"
: "${SMOLVLA_BASE:=lerobot/smolvla_base}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: image not found: ${APPTAINER_IMAGE}  (build it: bash env/build_image.sh)" >&2
  exit 1
fi

mkdir -p "${HF_HOME}"

# Apptainer module (so `apptainer` is on PATH on the login node).
if [[ -n "${APPTAINER_MODULE:-}" && "${APPTAINER_MODULE}" != "<"* ]]; then
  module load "${APPTAINER_MODULE}"
fi

echo "[aloha] dataset    : ${SMOLVLA_DATASET}"
echo "[aloha] base model : ${SMOLVLA_BASE}"
echo "[aloha] HF cache -> : ${HF_HOME}"
echo "[aloha] image      : ${APPTAINER_IMAGE}"

# Run hf / python from INSIDE the container so the right huggingface_hub + lerobot are
# guaranteed present. --home forces $HOME onto /work (small /home Lustre fs fills up,
# "Disk quota exceeded"); HF_HUB_DISABLE_XET=1 uses plain HTTPS (xet's growing chunk
# cache can overrun a quota mid-download). Same hardening as download_libero.sh.
APPTAINER_RUN=(apptainer exec
  --home "${HF_HOME}"
  --bind "${HF_HOME}:${HF_HOME}"
  --env "HF_HOME=${HF_HOME}"
  --env "HF_HUB_DISABLE_XET=1"
  "${APPTAINER_IMAGE}")

# --- 1. dataset into the HF cache (no --local-dir: resolved later by repo_id) ---
echo "== [1/3] downloading dataset =="
"${APPTAINER_RUN[@]}" hf download "${SMOLVLA_DATASET}" --repo-type dataset

# --- 2. SmolVLA base model into the HF cache --------------------------------
echo "== [2/3] downloading SmolVLA base model =="
"${APPTAINER_RUN[@]}" hf download "${SMOLVLA_BASE}"

# --- 3. warm-load the dataset once (online) so any conversion is cached here ---
echo "== [3/3] warm-loading the dataset (caches any format conversion) =="
if ! "${APPTAINER_RUN[@]}" python - "${SMOLVLA_DATASET}" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id = sys.argv[1]
ds = LeRobotDataset(repo_id)
print(f"[aloha] loaded OK: {ds.num_episodes} episodes, {ds.num_frames} frames")
PY
then
  echo "WARN: warm-load failed. The files are downloaded, but if offline training" >&2
  echo "      hits a dataset-version error, re-run this step online." >&2
fi

echo
echo "[aloha] done. Both the dataset and SmolVLA base are cached under ${HF_HOME}."
echo "  train it (batch, offline):"
echo "    source config.env && qsub -q \"\$QUEUE_NAME\" -W group_list=\"\$GROUP\" \\"
echo "      -l select=1 -l walltime=\"\$WALLTIME\" 04_policy_training/exercise/train_smolvla.pbs"
echo "  or interactively (see exercise/README.md, Answer A2)."
