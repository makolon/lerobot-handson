#!/usr/bin/env bash
# =============================================================================
# predownload_hf.sh  —  pre-download HF assets on the login node
# -----------------------------------------------------------------------------
# Why pre-download:
#   Miyabi [compute nodes are offline] (no external network access) by assumption.
#   So fetching from the HF Hub inside a training/eval job would hang / time out.
#   -> Cache into the shared HF_HOME on the login node (network OK), and have the
#      compute job use only that cache via HF_HUB_OFFLINE=1.
#
# Run on: the [login node].
# =============================================================================
set -euo pipefail

# --- fail-fast: required variables ---
: "${HF_HOME:?source config.env and set HF_HOME (shared area)}"
: "${DATA_REPO:?source config.env and set DATA_REPO}"
: "${CKPT_REPO:?source config.env and set CKPT_REPO}"

# Simple guard against running with placeholders still in place
case "${DATA_REPO}${CKPT_REPO}${HF_HOME}" in
  *"<TODO"*) echo "ERROR: config.env placeholders are not edited yet" >&2; exit 1;;
esac

export HF_HOME
mkdir -p "${HF_HOME}"
echo "[predl] HF_HOME = ${HF_HOME}"

# huggingface-cli is included in the assumed v0.5.1 env. If missing: `pip install -U huggingface_hub`.
# Pre-fetch the dataset (LeRobotDataset format)
echo "[predl] dataset: ${DATA_REPO}"
huggingface-cli download "${DATA_REPO}" --repo-type dataset

# Pre-fetch the distributed checkpoint used for evaluation
echo "[predl] ckpt   : ${CKPT_REPO}"
huggingface-cli download "${CKPT_REPO}" --repo-type model

echo "[predl] done. Compute nodes can now run offline with HF_HUB_OFFLINE=1."
echo "[predl] check size: du -sh ${HF_HOME}"
# TODO(miyabi): confirm the actual data/ckpt size and free space in the shared area beforehand.
