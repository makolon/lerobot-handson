#!/usr/bin/env bash
# =============================================================================
# train_smolvla.sh  —  ANSWER to Exercise Q2 (fine-tune SmolVLA on ALOHA Sim)
# -----------------------------------------------------------------------------
# Thin SmolVLA-specific config layer over ../train.sh (the single lerobot-train
# body). Same pattern as train_libero.sh: set SmolVLA-appropriate values, then
# exec the shared body, so there is one source of truth for the train command.
#
# Called from: exercise/train_smolvla.pbs (via apptainer exec) on Miyabi, OR run
#              directly inside an interactive container session.
# Prereq:      exercise/download_aloha_sim.sh has cached the dataset + the
#              lerobot/smolvla_base model into $HF_HOME on the login node.
#
# Key differences from ACT (train_libero.sh):
#   - fine-tunes a pretrained checkpoint: POLICY_PATH=lerobot/smolvla_base
#     (NOT --policy.type; ../train.sh switches on POLICY_PATH being set)
#   - SmolVLA is a VLA and reads the dataset's `task` language field
#   - larger than ACT -> small batch (4) + bfloat16 to fit memory
#   - dataset is resolved by repo_id from the HF cache (no DATASET_ROOT)
#
# NOTE on overrides: config.env exports generic defaults (TRAIN_STEPS, BATCH_SIZE,
# DATA_REPO, JOB_NAME) which Apptainer leaks into the container. To avoid silently
# inheriting those, this script uses dedicated SMOLVLA_* vars and *forces* the
# values it hands to ../train.sh. Override:
#   SMOLVLA_DATASET   dataset repo_id   (default lerobot/aloha_sim_insertion_human)
#   SMOLVLA_BASE      base checkpoint   (default lerobot/smolvla_base) -> --policy.path
#   SMOLVLA_DTYPE     compute dtype     (default bfloat16)
#   SMOLVLA_STEPS     train steps       (default 500 — a short "does it train?" run)
#   SMOLVLA_BATCH     batch size        (default 4 — SmolVLA is big; keep it small)
#   SMOLVLA_JOB_NAME  run/output name   (default smolvla_aloha)
#   OUTPUT_DIR        output root       (required; from config.env)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force the values ../train.sh consumes, from SMOLVLA_* overrides (NOT the generic
# config.env names, which leak in via Apptainer's env passthrough).
export DATA_REPO="${SMOLVLA_DATASET:-lerobot/aloha_sim_insertion_human}"
export POLICY_PATH="${SMOLVLA_BASE:-lerobot/smolvla_base}"   # fine-tune FROM this
export POLICY_DTYPE="${SMOLVLA_DTYPE:-bfloat16}"
export TRAIN_STEPS="${SMOLVLA_STEPS:-500}"
export BATCH_SIZE="${SMOLVLA_BATCH:-4}"
export JOB_NAME="${SMOLVLA_JOB_NAME:-smolvla_aloha}"

# Dataset comes from the HF cache (downloaded on the login node), resolved by
# repo_id with HF_HUB_OFFLINE=1 — so we deliberately do NOT set DATASET_ROOT.
unset DATASET_ROOT
# SmolVLA brings its own pretrained weights via POLICY_PATH, so the ACT-only
# ResNet18 backbone knob is irrelevant here.
unset PRETRAINED_BACKBONE_WEIGHTS

echo "[train_smolvla] dataset    : ${DATA_REPO}  (from HF cache)"
echo "[train_smolvla] fine-tune  : ${POLICY_PATH}  dtype=${POLICY_DTYPE}"

# Hand off to the shared training body (one dir up).
exec bash "${HERE}/../train.sh"
