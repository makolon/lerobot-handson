#!/usr/bin/env bash
# =============================================================================
# eval.sh  —  the lerobot-eval body (runs inside the container)
# -----------------------------------------------------------------------------
# Design intent:
#   By default it evaluates the [distributed checkpoint (CKPT_REPO)] in LIBERO.
#   That way you can see a success rate and feel a sense of accomplishment even if
#   your own training doesn't finish in time (no dependency on training completion).
#
# Confirmed minimal example (from the official README):
#   lerobot-eval --policy.path=lerobot/pi0_libero_finetuned \
#                --env.type=libero --env.task=libero_object --eval.n_episodes=10
#
# TODO(lerobot): confirm the exact spelling of --env.task options, --eval.batch_size,
#                --output_dir, --policy.device via `lerobot-eval --help` for v0.5.1.
# =============================================================================
set -euo pipefail

# --- fail-fast ---
: "${CKPT_REPO:?CKPT_REPO unset (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"
: "${EVAL_EPISODES:=10}"
: "${ENV_TASK:=libero_object}"   # TODO(lerobot): adjust to a task matching the distributed ckpt

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Checkpoint to evaluate. Default is the distributed repo.
# To evaluate your own training output, replace below with a path under OUTPUT_DIR:
#   POLICY_PATH="${OUTPUT_DIR}/${JOB_NAME}/checkpoints/last/pretrained_model"
POLICY_PATH="${CKPT_REPO}"

echo "[eval] policy=${POLICY_PATH} env=libero task=${ENV_TASK} episodes=${EVAL_EPISODES}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --env.type=libero \
  --env.task="${ENV_TASK}" \
  --eval.n_episodes="${EVAL_EPISODES}" \
  --output_dir="${OUTPUT_DIR}/eval_$(basename "${POLICY_PATH}")" \
  --policy.device=cuda

echo "[eval] done. See the output dir / log for the success rate and videos."
