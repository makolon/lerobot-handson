#!/usr/bin/env bash
# =============================================================================
# build_image.sh  —  build the Apptainer image on the login node
# -----------------------------------------------------------------------------
# Run on: the Miyabi [login node] (not a compute node).
# Note: the build is heavy (network/CPU). The assumption is to run it directly on
#       the login node rather than via qsub (but confirm the shared-usage policy).
#
# TODO(miyabi): confirm whether a long build is allowed on the login node, or
#               whether a dedicated build path (fakeroot/--remote) is required.
# =============================================================================
set -euo pipefail

# --- fail-fast: required variables ---
: "${APPTAINER_IMAGE:?source config.env and set APPTAINER_IMAGE}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${HERE}/apptainer.def"

# TODO(miyabi): confirm the module needed to use Apptainer/Singularity.
if [[ -n "${APPTAINER_MODULE:-}" && "${APPTAINER_MODULE}" != "<"* ]]; then
  module load "${APPTAINER_MODULE}"
fi

echo "[build] def : ${DEF_FILE}"
echo "[build] sif : ${APPTAINER_IMAGE}"

mkdir -p "$(dirname "${APPTAINER_IMAGE}")"

# Many environments require fakeroot. If you lack the privilege, consider --remote.
# TODO(miyabi): confirm fakeroot availability / whether --remote is required.
apptainer build --fakeroot "${APPTAINER_IMAGE}" "${DEF_FILE}"

echo "[build] done. test import:"
apptainer exec "${APPTAINER_IMAGE}" python -c "import lerobot; print('lerobot OK')"
