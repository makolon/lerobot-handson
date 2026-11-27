#!/usr/bin/env bash
# =============================================================================
# predownload_hf.sh  —  pre-stage a Hub checkpoint in the shared HF cache
# -----------------------------------------------------------------------------
# Why pre-stage:
#   Compute nodes have internet, but staging once into the shared HF_HOME means every
#   job reads the same files quickly and reproducibly, and no participant has to
#   download them (the shared cache is read-only for participants anyway).
#
# What it stages: CKPT_REPO (a policy checkpoint for §5), if set.
# What it does NOT stage: the LIBERO dataset. LeRobot reads datasets from a local root
#   ($LIBERO_ROOT, via --dataset.root) or from its own cache ($HF_HOME/lerobot/), NOT
#   from the generic Hub cache that `hf download` fills — so the dataset is staged by
#   04_policy_training/download_libero.sh instead.
#
# Run on: the [login node], as the organizer (after `source config.env`).
# =============================================================================
set -euo pipefail

# --- fail-fast: required variables ---
: "${HF_HOME:?source config.env and set HF_HOME (shared area)}"

# Simple guard against running with placeholders still in place
case "${CKPT_REPO:-}${HF_HOME}" in
  *"<TODO"*) echo "ERROR: config.env placeholders are not edited yet" >&2; exit 1;;
esac

export HF_HOME
mkdir -p "${HF_HOME}"
echo "[predl] HF_HOME = ${HF_HOME}"

# hf is included in the v0.5.1 env (huggingface_hub >= 1.0; `huggingface-cli` no longer
# exists). If missing on the host: run this inside the image, or `pip install -U huggingface_hub`.
if [[ -n "${CKPT_REPO:-}" ]]; then
  echo "[predl] ckpt   : ${CKPT_REPO}"
  hf download "${CKPT_REPO}" --repo-type model
else
  echo "[predl] CKPT_REPO is empty -> nothing to stage (participants evaluate their own checkpoint)."
fi

echo "[predl] dataset: staged by 04_policy_training/download_libero.sh into \$LIBERO_ROOT."
echo "[predl] done. check size: du -sh ${HF_HOME}"
