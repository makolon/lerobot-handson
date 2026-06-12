#!/usr/bin/env bash
# =============================================================================
# download_pretrained.sh  —  ANSWER to Exercise Q1 (fetch a pretrained checkpoint)
# -----------------------------------------------------------------------------
# Run on the Miyabi [LOGIN NODE] (it has internet; compute nodes are offline).
#
# Downloads ONE pretrained policy checkpoint (a MODEL repo) into the shared HF
# cache ($HF_HOME), so every compute job evaluates the SAME pre-staged files
# (faster + reproducible than each person re-downloading).
#
# Default: lerobot/act_aloha_sim_transfer_cube_human — an ACT policy on the ALOHA
# sim env. Chosen because it actually LOADS in this image: it needs no extra
# policy deps (ACT), and the aloha env ships in the image (lerobot[aloha]).
#
#   NOT every Hub checkpoint runs here. This image has lerobot[aloha,libero] only:
#     - ACT + aloha / ACT + libero        -> work
#     - diffusion_pusht (--env.type=pusht)-> needs gym-pusht  (NOT installed)
#     - pi0 / pi0.5 libero VLAs           -> need lerobot[pi] (NOT installed)
#   So pick an ACT checkpoint on aloha or libero, or evaluate your own Section 4
#   ACT-LIBERO checkpoint (no download needed).
#
# Override before running:
#   CKPT      model repo_id   (default lerobot/act_aloha_sim_transfer_cube_human)
#   HF_HOME   HF cache        (from config.env; the bind target)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

# config.env gives APPTAINER_IMAGE / APPTAINER_MODULE / HF_HOME.
if [[ -f "${REPO}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${REPO}/config.env"
fi

: "${APPTAINER_IMAGE:?APPTAINER_IMAGE unset — source config.env and build the image (01_hpc) first}"
: "${SHARE_DIR:=/work/gw13/share/handson}"
: "${HF_HOME:=${SHARE_DIR}/hf_home}"
: "${CKPT:=lerobot/act_aloha_sim_transfer_cube_human}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: image not found: ${APPTAINER_IMAGE}  (build it: bash env/build_image.sh)" >&2
  exit 1
fi

mkdir -p "${HF_HOME}"

if [[ -n "${APPTAINER_MODULE:-}" && "${APPTAINER_MODULE}" != "<"* ]]; then
  module load "${APPTAINER_MODULE}"
fi

echo "[ckpt] checkpoint : ${CKPT}"
echo "[ckpt] HF cache ->: ${HF_HOME}"
echo "[ckpt] image      : ${APPTAINER_IMAGE}"

# Run hf from INSIDE the container (right huggingface_hub). --home forces $HOME onto
# /work (small /home Lustre fs fills up); HF_HUB_DISABLE_XET=1 uses plain HTTPS. Same
# hardening as download_libero.sh / download_aloha_sim.sh.
echo "== downloading checkpoint =="
apptainer exec \
  --home "${HF_HOME}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --env "HF_HOME=${HF_HOME}" \
  --env "HF_HUB_DISABLE_XET=1" \
  "${APPTAINER_IMAGE}" \
  hf download "${CKPT}"

echo
echo "[ckpt] done. ${CKPT} is cached under ${HF_HOME}."
echo "  evaluate it (offline) with:  bash 05_policy_evaluation/exercise/eval_pretrained.sh"
