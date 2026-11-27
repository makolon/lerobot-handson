#!/usr/bin/env bash
# =============================================================================
# eval_pretrained.sh  —  ANSWER to Exercise Q2 (evaluate a pretrained checkpoint)
# -----------------------------------------------------------------------------
# Runs lerobot-eval on a Hub checkpoint in its matching simulator and prints a
# success rate. Mirrors ../eval.sh, but the checkpoint + env are parameterized so
# you can point it at the policy you downloaded in Q1.
#
# Called inside the container (interactive `apptainer shell --nv`, or via
# eval_pretrained.pbs). Prereq: download_pretrained.sh cached $CKPT into $USER_HF_HOME.
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
#   EVAL_OUT             output dir            (default $OUTPUT_DIR/eval/<checkpoint name>)
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

# This wrapper is driven by ENV VARS (see header), NOT by lerobot-eval flags. As a
# convenience it accepts a lerobot-style --policy.path=DIR (mapped to POLICY_PATH so the
# output dir is derived correctly); any other argument is rejected rather than silently
# ignored (which previously let a stale POLICY_PATH win over what you typed).
for a in "$@"; do
  case "$a" in
    --policy.path=*) POLICY_PATH="${a#--policy.path=}" ;;
    *) echo "ERROR: unrecognized argument: $a" >&2
       echo "  Set inputs via env vars (POLICY_PATH=... ENV_TYPE=... etc; see header)" >&2
       echo "  or pass --policy.path=DIR. This wrapper does not forward other lerobot-eval flags." >&2
       exit 2 ;;
  esac
done

# Checkpoint: default the well-known ALOHA-sim ACT policy (loads in this image).
POLICY_PATH="${POLICY_PATH:-${CKPT:-lerobot/act_aloha_sim_transfer_cube_human}}"

# Read the checkpoint from YOUR cache, where download_pretrained.sh put it.
if [[ -n "${USER_HF_HOME:-}" ]]; then export HF_HOME="${USER_HF_HOME}"; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Headless GPU rendering for the simulator (MuJoCo/robosuite). config.env may set
# these already; assert them for standalone runs.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

# Output dir, named after the RUN — not after the last path component, which is
# "pretrained_model" for every local checkpoint (so all runs used to overwrite one dir):
#   $OUTPUT_DIR/act_base/checkpoints/last/pretrained_model   -> $OUTPUT_DIR/eval/act_base
#   $OUTPUT_DIR/act_base/checkpoints/000500/pretrained_model -> $OUTPUT_DIR/eval/act_base_000500
#   lerobot/act_aloha_sim_transfer_cube_human                -> $OUTPUT_DIR/eval/act_aloha_sim_transfer_cube_human
# Set EVAL_OUT to override.
_ckpt="${POLICY_PATH%/}"
if [[ "$(basename "${_ckpt}")" == "pretrained_model" ]]; then
  _step="$(basename "$(dirname "${_ckpt}")")"
  _run="$(basename "$(dirname "$(dirname "$(dirname "${_ckpt}")")")")"
  if [[ "${_step}" != "last" ]]; then _run="${_run}_${_step}"; fi
else
  _run="$(basename "${_ckpt}")"
fi
EVAL_OUT="${EVAL_OUT:-${OUTPUT_DIR}/eval/${_run}}"

ARGS=(
  --policy.path="${POLICY_PATH}"
  --env.type="${ENV_TYPE}"
  --env.task="${ENV_TASK}"
  --eval.n_episodes="${EVAL_EPISODES}"
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --policy.device="${POLICY_DEVICE}"
  --output_dir="${EVAL_OUT}"
)
# lerobot-eval always records rollout mp4s (up to 10 per task) under
# ${EVAL_OUT}/videos/<task_group>_<task_id>/ — there is no CLI flag to toggle it in lerobot 0.5.1.

echo "[eval] policy=${POLICY_PATH} env=${ENV_TYPE} task=${ENV_TASK} episodes=${EVAL_EPISODES}"
echo "[eval] lerobot-eval ${ARGS[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[eval] DRY_RUN=1 -> command construction verified, not executing."
  exit 0
fi

lerobot-eval "${ARGS[@]}"

echo "[eval] done. success rate -> the log + ${EVAL_OUT}/eval_info.json"
echo "[eval] rollout videos (up to 10 per task) -> ${EVAL_OUT}/videos/"
