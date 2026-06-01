#!/usr/bin/env bash
# =============================================================================
# run_tuning.sh  —  runner for the short tuning competition (Bonus 2)
# -----------------------------------------------------------------------------
# [Assumes an interactive node]: run directly on a reserved interactive/debug node,
#   not via batch submission (reserve an interactive node -> enter the container or
#   apptainer exec -> run this script).
#   TODO(miyabi): confirm how to reserve an interactive node (qsub -I, etc.).
#
# Fixed for fairness: step count / walltime / dataset / policy type.
# Knobs (arguments or environment variables) mapped to real lerobot-train flags
# (verified with lerobot==0.5.1):
#   CHUNK_SIZE -> --policy.chunk_size   (action horizon)
#   OBS_STEPS  -> --policy.n_obs_steps  (observation window)
#   LR         -> --policy.optimizer_lr (learning rate)
#   BATCH_SIZE -> --batch_size
#   AUG=on     -> --dataset.image_transforms.enable=true   (image augmentation)
#
# Also exercised by tools/smoke_test.sh on CPU (DATASET_ROOT + POLICY_DEVICE=cpu +
# PRETRAINED_BACKBONE_WEIGHTS=null + tiny FIXED_STEPS).
# =============================================================================
set -euo pipefail

# --- fixed values (do not change, for fairness) ---
FIXED_STEPS="${FIXED_STEPS:-1000}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

# --- fail-fast: dataset/output are required ---
: "${DATA_REPO:?DATA_REPO unset (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"

# --- knobs (with defaults) ---
CHUNK_SIZE="${CHUNK_SIZE:-50}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OBS_STEPS="${OBS_STEPS:-1}"
AUG="${AUG:-off}"

# Embed the knobs in the run name to tell them apart in W&B
RUN_NAME="lb_${POLICY_TYPE}_cs${CHUNK_SIZE}_lr${LR}_bs${BATCH_SIZE}_obs${OBS_STEPS}_aug${AUG}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ARGS=(
  --policy.type="${POLICY_TYPE}"
  --dataset.repo_id="${DATA_REPO}"
  --steps="${FIXED_STEPS}"
  --batch_size="${BATCH_SIZE}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
  --policy.n_obs_steps="${OBS_STEPS}"
  --policy.optimizer_lr="${LR}"
  --output_dir="${OUTPUT_DIR}/${RUN_NAME}"
  --job_name="${RUN_NAME}"
  --policy.device="${POLICY_DEVICE}"
  --policy.push_to_hub=false
)

# Local dataset for offline/smoke runs (on Miyabi the dataset comes from HF_HOME).
[[ -n "${DATASET_ROOT:-}" ]] && ARGS+=( --dataset.root="${DATASET_ROOT}" )
# Skip the ResNet ImageNet download when requested (offline smoke).
[[ -n "${PRETRAINED_BACKBONE_WEIGHTS:-}" ]] && \
  ARGS+=( --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}" )

# Image augmentation knob
[[ "${AUG}" == "on" ]] && ARGS+=( --dataset.image_transforms.enable=true )

# Shared W&B is what makes this a leaderboard. Without a project, warn and run with W&B off.
if [[ -n "${WANDB_PROJECT:-}" && "${WANDB_PROJECT}" != "<"* ]]; then
  ARGS+=( --wandb.enable=true --wandb.project="${WANDB_PROJECT}" )
  [[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "<"* ]] && ARGS+=( --wandb.entity="${WANDB_ENTITY}" )
else
  echo "[tuning] WARNING: WANDB_PROJECT unset -> running with W&B disabled; this run will NOT appear on the leaderboard." >&2
  ARGS+=( --wandb.enable=false )
fi

echo "[tuning] ${RUN_NAME}"
echo "[tuning] lerobot-train ${ARGS[*]}"

# Inside the container on the interactive node, call lerobot-train directly.
# From outside, wrap with: apptainer exec --nv "${APPTAINER_IMAGE}" ...
lerobot-train "${ARGS[@]}"

echo "[tuning] done. Check the score for ${RUN_NAME} in the shared W&B (${WANDB_PROJECT:-<disabled>})."
