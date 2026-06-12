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
export POLICY_TYPE="smolvla"          # train SmolVLA (see note on why NOT --policy.path)
export TRAIN_STEPS="${SMOLVLA_STEPS:-500}"
export BATCH_SIZE="${SMOLVLA_BATCH:-4}"
export JOB_NAME="${SMOLVLA_JOB_NAME:-smolvla_aloha}"

# Why --policy.type=smolvla (+ load the pretrained VLM) and NOT --policy.path=smolvla_base:
#   smolvla_base's saved config hard-codes 3 cameras (camera1/2/3). An ALOHA dataset has a
#   single 'top' camera, so --policy.path=lerobot/smolvla_base errors with a feature
#   mismatch. --policy.type=smolvla derives the input features FROM THE DATASET and loads
#   the pretrained SmolVLM2 vision-language backbone (load_vlm_weights=true) while training
#   a fresh action expert — so it trains end-to-end on any dataset's camera layout.
export EXTRA_ARGS="--policy.load_vlm_weights=true ${EXTRA_ARGS:-}"
unset POLICY_PATH

# num2words (a SmolVLM-processor dependency) is staged in the shared pylibs dir rather than
# baked into the image; put it on PYTHONPATH. (Rebuild with lerobot[...,smolvla] to drop this.)
export PYTHONPATH="${SMOLVLA_PYLIBS:-${SHARED_DIR:-/work/gw13/share/handson}/pylibs}:${PYTHONPATH:-}"

# Dataset is resolved by repo_id from the HF cache (compute nodes have internet) — so we
# deliberately do NOT set DATASET_ROOT. ResNet backbone knob is ACT-only, so clear it.
unset DATASET_ROOT
unset PRETRAINED_BACKBONE_WEIGHTS
# SmolVLAConfig has no `dtype` field (unlike ACT), so DON'T pass --policy.dtype (draccus
# rejects it). On a GH200 (96 GB) fp32 fits fine here; SmolVLA manages its backbone precision.
unset POLICY_DTYPE

echo "[train_smolvla] dataset : ${DATA_REPO}  (from HF cache)"
echo "[train_smolvla] policy  : smolvla (pretrained SmolVLM2 backbone + fresh action expert)"
echo "[train_smolvla] pylibs  : ${PYTHONPATH%%:*}  (num2words)"

# Hand off to the shared training body (one dir up).
exec bash "${HERE}/../train.sh"
