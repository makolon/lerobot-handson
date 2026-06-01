#!/usr/bin/env bash
# =============================================================================
# train.sh  —  the lerobot-train body (runs inside the container)
# -----------------------------------------------------------------------------
# Called from: 03_train/train.pbs (via apptainer exec).
# Can also be tested standalone: inside the container, `bash 03_train/train.sh`.
#
# Fallback design:
#   The default is "lightweight policy (ACT) + few steps (TRAIN_STEPS)" so it always
#   "flows" within 105 minutes (loss moves, a checkpoint is produced). For serious
#   training, increase TRAIN_STEPS / switch to a heavier policy.
#
# TODO(lerobot): confirm lerobot-train arg names via `lerobot-train --help` for v0.5.1.
#   Confirmed minimal example (from the official README):
#     lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
#   The --batch_size / --steps / --output_dir / --policy.device / --wandb.* below are
#   draccus-derived config keys, but their exact spelling needs confirmation.
# =============================================================================
set -euo pipefail

# --- fail-fast: required variables ---
: "${DATA_REPO:?DATA_REPO unset (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"
: "${POLICY_TYPE:=act}"
: "${TRAIN_STEPS:=2000}"
: "${BATCH_SIZE:=8}"
: "${JOB_NAME:=handson_${POLICY_TYPE}}"

# W&B (shared). Kept loose so it still works when unset.
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" && "${WANDB_PROJECT}" != "<"* ]]; then
  # TODO(lerobot): confirm the wandb flags (--wandb.enable / --wandb.project /
  #                --wandb.entity) for v0.5.1.
  WANDB_ARGS+=( "--wandb.enable=true" "--wandb.project=${WANDB_PROJECT}" )
  [[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "<"* ]] && \
    WANDB_ARGS+=( "--wandb.entity=${WANDB_ENTITY}" )
fi

echo "[train] policy=${POLICY_TYPE} dataset=${DATA_REPO} steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"
echo "[train] output=${OUTPUT_DIR}/${JOB_NAME}"

# Compute nodes are offline. train.pbs already exports this; re-assert for standalone runs.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

lerobot-train \
  --policy.type="${POLICY_TYPE}" \
  --dataset.repo_id="${DATA_REPO}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${TRAIN_STEPS}" \
  --output_dir="${OUTPUT_DIR}/${JOB_NAME}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  "${WANDB_ARGS[@]}"

# --- Fallback option (an alternative way to reliably "flow" in a short time) ---
# To short fine-tune from an existing distributed checkpoint, start from a pretrained
# policy instead of --policy.type above:
#   TODO(lerobot): confirm how to start from a pretrained checkpoint for v0.5.1
#                  (e.g. --policy.path=${CKPT_REPO}).
echo "[train] done. checkpoints -> ${OUTPUT_DIR}/${JOB_NAME}"
