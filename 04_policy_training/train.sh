#!/usr/bin/env bash
# =============================================================================
# train.sh  —  the lerobot-train body (runs inside the container)
# -----------------------------------------------------------------------------
# Called from: 03_train/train.pbs (via apptainer exec) on Miyabi.
# Also exercised directly by tools/smoke_test.sh on CPU (no GPU/net).
#
# Real lerobot-train invocation (verified with lerobot==0.5.1). Everything here is
# bucket-2 (runs anywhere); only train.pbs carries bucket-1 #PBS placeholders.
#
# Fallback design: the default is "lightweight policy (ACT) + few steps" so a run
# always "flows" within 105 min (loss moves, a checkpoint is written). Increase
# TRAIN_STEPS or pass a heavier POLICY_TYPE for serious training.
#
# Overridable environment variables (config.env supplies the Miyabi defaults):
#   DATA_REPO        repo_id of the dataset                (required)
#   DATASET_ROOT     local dataset dir; if set, adds --dataset.root (offline/local)
#   OUTPUT_DIR       output root                            (required)
#   POLICY_TYPE      act | diffusion | ...                  (default act; fresh policy)
#   POLICY_PATH      pretrained checkpoint to fine-tune from (optional; if set, uses
#                    --policy.path INSTEAD of --policy.type, e.g. lerobot/smolvla_base)
#   POLICY_DTYPE     float32 | bfloat16 | ...               (optional; e.g. VLA fine-tune)
#   TRAIN_STEPS      number of train steps                  (default 2000)
#   BATCH_SIZE       batch size                             (default 8)
#   POLICY_DEVICE    cuda | cpu                             (default cuda)
#   JOB_NAME         run/output name                        (default handson_<policy>)
#   CHUNK_SIZE / N_OBS_STEPS / N_ACTION_STEPS               (optional; added if set)
#   PRETRAINED_BACKBONE_WEIGHTS  e.g. "null" to skip the ImageNet download (optional)
#   SAVE_FREQ / LOG_FREQ                                    (optional)
# =============================================================================
set -euo pipefail

# --- fail-fast: required variables ---
: "${DATA_REPO:?DATA_REPO unset (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"
: "${POLICY_TYPE:=act}"
: "${TRAIN_STEPS:=2000}"
: "${BATCH_SIZE:=8}"
: "${POLICY_DEVICE:=cuda}"
: "${JOB_NAME:=handson_${POLICY_TYPE}}"
: "${SAVE_FREQ:=${TRAIN_STEPS}}"
: "${LOG_FREQ:=100}"

ARGS=(
  --dataset.repo_id="${DATA_REPO}"
  --batch_size="${BATCH_SIZE}"
  --steps="${TRAIN_STEPS}"
  --output_dir="${OUTPUT_DIR}/${JOB_NAME}"
  --job_name="${JOB_NAME}"
  --policy.device="${POLICY_DEVICE}"
  --policy.push_to_hub=false      # we share via W&B, not the Hub; avoids needing policy.repo_id
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"
)

# Policy source: fine-tune a pretrained checkpoint (--policy.path) when POLICY_PATH is
# set, otherwise start a fresh policy of --policy.type. They are mutually exclusive in
# lerobot-train, so we pick exactly one.
if [[ -n "${POLICY_PATH:-}" ]]; then
  ARGS+=( --policy.path="${POLICY_PATH}" )
  POLICY_DESC="path:${POLICY_PATH}"
else
  ARGS+=( --policy.type="${POLICY_TYPE}" )
  POLICY_DESC="type:${POLICY_TYPE}"
fi

# Optional dtype (e.g. bfloat16 to fit a VLA fine-tune in memory).
[[ -n "${POLICY_DTYPE:-}" ]] && ARGS+=( --policy.dtype="${POLICY_DTYPE}" )

# Local dataset (offline / synthetic). On Miyabi the dataset is resolved from HF_HOME.
[[ -n "${DATASET_ROOT:-}" ]] && ARGS+=( --dataset.root="${DATASET_ROOT}" )

# Optional policy knobs (only added when set)
[[ -n "${CHUNK_SIZE:-}" ]]      && ARGS+=( --policy.chunk_size="${CHUNK_SIZE}" )
[[ -n "${N_OBS_STEPS:-}" ]]     && ARGS+=( --policy.n_obs_steps="${N_OBS_STEPS}" )
[[ -n "${N_ACTION_STEPS:-}" ]]  && ARGS+=( --policy.n_action_steps="${N_ACTION_STEPS}" )
# "null" skips the ResNet ImageNet download (used by the offline smoke test)
[[ -n "${PRETRAINED_BACKBONE_WEIGHTS:-}" ]] && \
  ARGS+=( --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}" )

# W&B (shared). Enabled only when a project is configured.
if [[ -n "${WANDB_PROJECT:-}" && "${WANDB_PROJECT}" != "<"* ]]; then
  ARGS+=( --wandb.enable=true --wandb.project="${WANDB_PROJECT}" )
  [[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "<"* ]] && ARGS+=( --wandb.entity="${WANDB_ENTITY}" )
else
  ARGS+=( --wandb.enable=false )
fi

# Compute nodes are offline; train.pbs exports this, re-assert for standalone runs.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[train] policy=${POLICY_DESC} dataset=${DATA_REPO} steps=${TRAIN_STEPS} batch=${BATCH_SIZE} device=${POLICY_DEVICE}"
echo "[train] output=${OUTPUT_DIR}/${JOB_NAME}"
echo "[train] lerobot-train ${ARGS[*]}"

lerobot-train "${ARGS[@]}"

# Alternative (real): short fine-tune from an existing distributed checkpoint by
# loading it with --policy.path=<repo_id_or_dir> instead of --policy.type=<...>.
echo "[train] done. checkpoints -> ${OUTPUT_DIR}/${JOB_NAME}"
