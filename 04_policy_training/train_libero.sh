#!/usr/bin/env bash
# =============================================================================
# train_libero.sh  —  train ACT on the SHARED LIBERO dataset (runs in container)
# -----------------------------------------------------------------------------
# Thin LIBERO-specific config layer over train.sh (the single lerobot-train body).
# It sets LIBERO-appropriate values and execs train.sh, so there is one source of
# truth for the actual training command.
#
# Called from: 04_policy_training/train_libero.pbs (via apptainer exec) on Miyabi.
# Prereq:      04_policy_training/download_libero.sh has populated $LIBERO_ROOT
#              (the share) and pre-cached the ResNet18 backbone in $TORCH_HOME.
#
# NOTE on overrides: config.env exports generic training defaults (TRAIN_STEPS=2000,
# BATCH_SIZE=8, DATA_REPO=<...>, JOB_NAME=...) which Apptainer passes into the
# container. To avoid silently inheriting those, this script uses dedicated
# LIBERO_* override vars and *forces* the values it hands to train.sh. Override:
#   SHARE_DIR        shared root            (default /work/gw13/share/handson)
#   LIBERO_ROOT      local dataset dir      (default $SHARE_DIR/libero) -> --dataset.root
#   LIBERO_REPO      dataset repo_id        (default HuggingFaceVLA/libero)
#   LIBERO_POLICY    act | diffusion | ...  (default act)
#   LIBERO_STEPS     train steps            (default 50000; published ACT runs use more)
#   LIBERO_BATCH     batch size             (default 32; a GH200's ~96 GB can go higher)
#   LIBERO_SAVE_FREQ checkpoint interval    (default 10000)
#   LIBERO_JOB_NAME  run/output name        (default libero_act)
#   OUTPUT_DIR       output root            (required; from config.env)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SHARE_DIR:=/work/gw13/share/handson}"

# Force the values train.sh consumes, from LIBERO_* overrides (NOT the generic
# config.env names, which leak in via Apptainer's env passthrough).
export DATA_REPO="${LIBERO_REPO:-HuggingFaceVLA/libero}"
export DATASET_ROOT="${LIBERO_ROOT:-${SHARE_DIR}/libero}"
export POLICY_TYPE="${LIBERO_POLICY:-act}"
export TRAIN_STEPS="${LIBERO_STEPS:-50000}"
export BATCH_SIZE="${LIBERO_BATCH:-32}"
export SAVE_FREQ="${LIBERO_SAVE_FREQ:-10000}"
export JOB_NAME="${LIBERO_JOB_NAME:-libero_act}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset not found at DATASET_ROOT=${DATASET_ROOT}" >&2
  echo "       Run 04_policy_training/download_libero.sh on the login node first." >&2
  exit 1
fi

echo "[train_libero] dataset root  : ${DATASET_ROOT}"
echo "[train_libero] backbone cache: ${TORCH_HOME:-<unset> (ACT will try to download ResNet18 — offline-unsafe)}"

# Hand off to the shared training body. ACT keeps its default ResNet18 ImageNet
# backbone (pre-cached in TORCH_HOME by download_libero.sh). To skip that download
# entirely (random init), export PRETRAINED_BACKBONE_WEIGHTS=null before calling.
exec bash "${HERE}/train.sh"
