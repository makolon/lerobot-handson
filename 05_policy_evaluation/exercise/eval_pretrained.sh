#!/usr/bin/env bash
# =============================================================================
# eval_pretrained.sh  —  ANSWER to Exercise Q2 (evaluate a pretrained checkpoint)
# -----------------------------------------------------------------------------
# Runs lerobot-eval on a Hub checkpoint in its matching simulator and prints a
# success rate. Mirrors ../eval.sh, but the checkpoint + env are parameterized so
# you can point it at the policy you downloaded in Q1.
#
# Called inside the container (interactive `apptainer shell --nv`, or via
# eval_pretrained.pbs). Prereq: download_pretrained.sh cached $CKPT into $HF_HOME.
#
# The checkpoint and the env MUST match — a LIBERO policy cannot run on the aloha
# env (the observation/action spaces differ). Defaults below pick a combination
# that LOADS in this image (ACT + aloha; see download_pretrained.sh for why).
#
# Overridable env vars:
#   CKPT / POLICY_PATH   checkpoint to evaluate (repo_id or local dir)
#   ENV_TYPE             aloha | libero        (default aloha)
#   ENV_TASK             task name             (default AlohaTransferCube-v0)
#   EVAL_EPISODES        number of episodes    (default 10)
#   EVAL_BATCH_SIZE      parallel envs         (default = EVAL_EPISODES)
#   POLICY_DEVICE        cuda | cpu            (default cuda)
#   OUTPUT_DIR           output root           (required; from config.env)
#   DRY_RUN              1 = print the command, don't run
# =============================================================================
set -euo pipefail

# --- fail-fast ---
: "${OUTPUT_DIR:?OUTPUT_DIR unset}"
: "${ENV_TYPE:=aloha}"
: "${ENV_TASK:=AlohaTransferCube-v0}"
: "${EVAL_EPISODES:=10}"
: "${EVAL_BATCH_SIZE:=${EVAL_EPISODES}}"
: "${POLICY_DEVICE:=cuda}"

# Checkpoint: default the well-known ALOHA-sim ACT policy (loads in this image).
POLICY_PATH="${POLICY_PATH:-${CKPT:-lerobot/act_aloha_sim_transfer_cube_human}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Headless GPU rendering for the simulator (MuJoCo/robosuite). config.env may set
# these already; assert them for standalone runs.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

EVAL_OUT="${OUTPUT_DIR}/eval_$(basename "${POLICY_PATH}")"

ARGS=(
  --policy.path="${POLICY_PATH}"
  --env.type="${ENV_TYPE}"
  --env.task="${ENV_TASK}"
  --env.video=true                 # save a rollout mp4 per episode (watch successes & failures)
  --eval.n_episodes="${EVAL_EPISODES}"
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --policy.device="${POLICY_DEVICE}"
  --output_dir="${EVAL_OUT}"
)
# Optional: cap the rollout video length in frames (default records the full episode).
[[ -n "${EVAL_VIDEO_LENGTH:-}" ]] && ARGS+=( --env.video_length="${EVAL_VIDEO_LENGTH}" )

echo "[eval] policy=${POLICY_PATH} env=${ENV_TYPE} task=${ENV_TASK} episodes=${EVAL_EPISODES}"
echo "[eval] lerobot-eval ${ARGS[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[eval] DRY_RUN=1 -> command construction verified, not executing."
  exit 0
fi

lerobot-eval "${ARGS[@]}"

echo "[eval] done. success rate -> the log + ${EVAL_OUT}/eval_info.json"
echo "[eval] rollout videos (one mp4 per episode) -> ${EVAL_OUT}/videos/"
