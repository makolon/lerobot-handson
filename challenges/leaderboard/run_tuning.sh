#!/usr/bin/env bash
# =============================================================================
# run_tuning.sh  —  runner for the short tuning competition (Bonus 2)
# -----------------------------------------------------------------------------
# [Assumes an interactive node]: run directly on a reserved interactive/debug node,
#   not via batch submission. (e.g. reserve an interactive node -> enter the container
#   or apptainer exec -> run this script)
#   TODO(miyabi): confirm how to reserve an interactive node (qsub -I, etc.).
#
# Fixed for fairness: step count / walltime / dataset / policy type.
# Knobs you can turn (arguments or environment variables):
#   CHUNK_SIZE : action horizon (number of action steps predicted at once)
#   LR         : learning rate
#   BATCH_SIZE : batch size
#   OBS_STEPS  : observation time window (observation steps)
#   AUG        : image augmentation on/off
#
# TODO(lerobot): confirm the lerobot-train config keys for each knob via
#   `lerobot-train --help` for v0.5.1. The spellings below are guesses based on the
#   common names for ACT-family policies:
#     chunk size  -> --policy.chunk_size (may be linked with --policy.n_action_steps)
#     obs steps   -> --policy.n_obs_steps
#     lr          -> --optimizer.lr or --policy.optimizer_lr
#     aug         -> the image-transforms enable flag (dataset.image_transforms.enable, etc.)
# =============================================================================
set -euo pipefail

# --- fixed values (do not change, for fairness) ---
FIXED_STEPS="${FIXED_STEPS:-1000}"
POLICY_TYPE="${POLICY_TYPE:-act}"

# --- fail-fast: without a shared W&B there is no leaderboard ---
: "${DATA_REPO:?DATA_REPO unset (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"
: "${WANDB_PROJECT:?WANDB_PROJECT unset: without the shared W&B the ranking is invisible}"

# --- knobs (with defaults) ---
CHUNK_SIZE="${CHUNK_SIZE:-50}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OBS_STEPS="${OBS_STEPS:-1}"
AUG="${AUG:-off}"

# Embed the knobs in the run name to tell them apart in W&B
RUN_NAME="lb_${POLICY_TYPE}_cs${CHUNK_SIZE}_lr${LR}_bs${BATCH_SIZE}_obs${OBS_STEPS}_aug${AUG}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Assemble the augmentation flag
AUG_ARGS=()
if [[ "${AUG}" == "on" ]]; then
  # TODO(lerobot): confirm the image-aug enable key for v0.5.1
  AUG_ARGS+=( "--dataset.image_transforms.enable=true" )
fi

WANDB_ARGS=( "--wandb.enable=true" "--wandb.project=${WANDB_PROJECT}" )
[[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "<"* ]] && \
  WANDB_ARGS+=( "--wandb.entity=${WANDB_ENTITY}" )

echo "[tuning] ${RUN_NAME}"

# If you're already inside the container on the interactive node, call lerobot-train directly.
# To run from outside the container, wrap the body with apptainer exec --nv "${APPTAINER_IMAGE}".
lerobot-train \
  --policy.type="${POLICY_TYPE}" \
  --dataset.repo_id="${DATA_REPO}" \
  --steps="${FIXED_STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_obs_steps="${OBS_STEPS}" \
  --optimizer.lr="${LR}" \
  --output_dir="${OUTPUT_DIR}/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --policy.device=cuda \
  "${AUG_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

echo "[tuning] done. Check the score for ${RUN_NAME} in the shared W&B (${WANDB_PROJECT})."
