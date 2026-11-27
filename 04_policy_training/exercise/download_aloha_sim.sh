#!/usr/bin/env bash
# =============================================================================
# download_aloha_sim.sh  —  ANSWER to Exercise Q1 (fetch dataset + SmolVLM2 backbone)
# -----------------------------------------------------------------------------
# Run on the Miyabi [LOGIN NODE]. (Compute nodes have internet too; downloading once
# here means the training job starts from a complete, fixed copy.)
#
# Downloads, into YOUR HF cache ($USER_HF_HOME = $USER_DIR/hf_home — the shared
# $HF_HOME is read-only for participants), the two things the SmolVLA run needs:
#   1. an ALOHA Sim dataset (default lerobot/aloha_sim_insertion_human). It is fetched
#      by LOADING it once with LeRobotDataset: LeRobot 0.5.1 reads datasets from its own
#      cache ($HF_HOME/lerobot/), not from the Hub cache that `hf download` fills.
#   2. SmolVLA's pretrained vision-language backbone,
#      HuggingFaceTB/SmolVLM2-500M-Video-Instruct (weights + processor). The answer trains
#      --policy.type=smolvla --policy.load_vlm_weights=true, which starts from it.
#      (lerobot/smolvla_base is NOT used: its config expects 3 cameras; ALOHA has 1.)
#
# After this, train with exercise/train_smolvla.{sh,pbs}.
#
# Override before running:
#   SMOLVLA_DATASET   HF dataset repo_id    (default lerobot/aloha_sim_insertion_human)
#   SMOLVLA_VLM       VLM backbone repo_id  (default HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
#   USER_HF_HOME      your HF cache         (from config.env; default $USER_DIR/hf_home)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

# config.env gives APPTAINER_IMAGE / APPTAINER_MODULE / USER_DIR / USER_HF_HOME.
if [[ -f "${REPO}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO}/config.env"
fi

: "${APPTAINER_IMAGE:?APPTAINER_IMAGE unset — source config.env and build the image (01_hpc) first}"
# Participant downloads go to the personal cache (the shared HF_HOME is read-only).
HF_HOME="${USER_HF_HOME:-${USER_DIR:?USER_DIR unset (source config.env)}/hf_home}"
: "${HF_DATASETS_CACHE:=${USER_DIR}/hf_datasets}"
: "${SMOLVLA_DATASET:=lerobot/aloha_sim_insertion_human}"
: "${SMOLVLA_VLM:=HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: image not found: ${APPTAINER_IMAGE}  (build it: bash env/build_image.sh)" >&2
  exit 1
fi

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# Apptainer module (so `apptainer` is on PATH on the login node).
if [[ -n "${APPTAINER_MODULE:-}" && "${APPTAINER_MODULE}" != "<"* ]]; then
  module load "${APPTAINER_MODULE}"
fi

echo "[aloha] dataset    : ${SMOLVLA_DATASET}"
echo "[aloha] VLM        : ${SMOLVLA_VLM}"
echo "[aloha] HF cache -> : ${HF_HOME}"
echo "[aloha] image      : ${APPTAINER_IMAGE}"

# Run hf / python from INSIDE the container so the right huggingface_hub + lerobot are
# guaranteed present. --home forces $HOME onto /work (small /home Lustre fs fills up,
# "Disk quota exceeded"); HF_HUB_DISABLE_XET=1 uses plain HTTPS (xet's growing chunk
# cache can overrun a quota mid-download). Only $PWD/$HOME are auto-mounted, so the
# caches are bound explicitly. Same hardening as download_libero.sh.
APPTAINER_RUN=(apptainer exec
  --home "${HF_HOME}"
  --bind "${HF_HOME}:${HF_HOME}"
  --bind "${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE}"
  --env "HF_HOME=${HF_HOME}"
  --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
  --env "HF_HUB_DISABLE_XET=1"
  "${APPTAINER_IMAGE}")

# --- 1. dataset: load it once, so it lands in LeRobot's cache ($HF_HOME/lerobot/) ---
echo "== [1/2] downloading the dataset (by loading it once with LeRobotDataset) =="
"${APPTAINER_RUN[@]}" python - "${SMOLVLA_DATASET}" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id = sys.argv[1]
ds = LeRobotDataset(repo_id)
print(f"[aloha] loaded OK: {ds.num_episodes} episodes, {ds.num_frames} frames")
PY

# --- 2. SmolVLA's VLM backbone (weights + processor) -> $HF_HOME/hub ----------
echo "== [2/2] downloading the SmolVLM2 backbone =="
"${APPTAINER_RUN[@]}" hf download "${SMOLVLA_VLM}"

echo
echo "[aloha] done. The dataset and the SmolVLM2 backbone are cached under ${HF_HOME}."
echo "  train it (batch):"
echo "    source config.env && qsub -q \"\$QUEUE_NAME\" -W group_list=\"\$GROUP\" \\"
echo "      -l select=1 -l walltime=\"\$WALLTIME\" 04_policy_training/exercise/train_smolvla.pbs"
echo "  or interactively (see exercise/README.md, Answer A2)."
