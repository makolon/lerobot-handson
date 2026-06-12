#!/usr/bin/env bash
# =============================================================================
# eval.sh  —  the lerobot-eval body (runs inside the container)
# -----------------------------------------------------------------------------
# Design intent:
#   By default it evaluates the [distributed checkpoint (CKPT_REPO)] in LIBERO,
#   so you get a success rate even if your own training doesn't finish in time.
#
# Real lerobot-eval invocation (verified with lerobot==0.5.1):
#   lerobot-eval --policy.path=<repo_or_dir> --env.type=libero \
#                --env.task=libero_object --eval.n_episodes=10 --eval.batch_size=10
# Note: eval.batch_size must be <= eval.n_episodes (lerobot raises otherwise).
#
# LIBERO needs simulation deps that may not be present on every machine, so set
#   DRY_RUN=1  to print the exact command without executing it (handy to verify the
#   command construction off-Miyabi). Set POLICY_DEVICE=cpu to evaluate on CPU.
#
# Overridable env vars:
#   CKPT_REPO / POLICY_PATH   checkpoint to evaluate (repo_id or local dir)
#   OUTPUT_DIR                output root
#   EVAL_EPISODES             number of eval episodes      (default 10)
#   EVAL_BATCH_SIZE           parallel envs                (default = EVAL_EPISODES)
#   ENV_TASK                  libero task name             (default libero_object)
#   POLICY_DEVICE             cuda | cpu                   (default cuda)
#   DRY_RUN                   1 = print the command, don't run
# =============================================================================
set -euo pipefail

# --- fail-fast ---
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"
: "${EVAL_EPISODES:=10}"
: "${EVAL_BATCH_SIZE:=${EVAL_EPISODES}}"
: "${ENV_TASK:=libero_object}"
: "${POLICY_DEVICE:=cuda}"

# Checkpoint to evaluate. Default is the distributed repo (CKPT_REPO).
# To evaluate your own training output, set:
#   POLICY_PATH="${OUTPUT_DIR}/${JOB_NAME}/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${CKPT_REPO:?set CKPT_REPO or POLICY_PATH}}"

# Compute nodes have internet; default ONLINE (set HF_HUB_OFFLINE=1 to force cache-only).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# Headless GPU rendering for the LIBERO simulator. Miyabi sets CUDA_VISIBLE_DEVICES to
# a GPU *UUID*, which robosuite's EGL device picker can't parse as an int; it prefers
# MUJOCO_EGL_DEVICE_ID, so pin a numeric EGL index (each Miyabi-G node = 1 GPU).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

ARGS=(
  --policy.path="${POLICY_PATH}"
  --env.type=libero
  --env.task="${ENV_TASK}"
  --eval.n_episodes="${EVAL_EPISODES}"
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --output_dir="${OUTPUT_DIR}/eval_$(basename "${POLICY_PATH}")"
  --policy.device="${POLICY_DEVICE}"
)

echo "[eval] policy=${POLICY_PATH} env=libero task=${ENV_TASK} episodes=${EVAL_EPISODES}"
echo "[eval] lerobot-eval ${ARGS[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[eval] DRY_RUN=1 -> command construction verified, not executing."
  exit 0
fi

lerobot-eval "${ARGS[@]}"

echo "[eval] done. See the output dir / log for the success rate and videos."
