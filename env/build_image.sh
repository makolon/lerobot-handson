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

: "${APPTAINER_IMAGE:?source config.env and set APPTAINER_IMAGE}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${HERE}/apptainer.def"

if [[ -n "${APPTAINER_MODULE:-}" && "${APPTAINER_MODULE}" != "<"* ]]; then
  module load "${APPTAINER_MODULE}"
fi

: "${APPTAINER_TMPDIR:=/tmp/${USER}-aptmp}"
: "${APPTAINER_CACHEDIR:=/tmp/${USER}-apcache}"
export APPTAINER_TMPDIR APPTAINER_CACHEDIR
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "[build] def      : ${DEF_FILE}"
echo "[build] sif      : ${APPTAINER_IMAGE}"
echo "[build] tmpdir   : ${APPTAINER_TMPDIR}"
echo "[build] cachedir : ${APPTAINER_CACHEDIR}"

mkdir -p "$(dirname "${APPTAINER_IMAGE}")"

apptainer build --fakeroot "${APPTAINER_IMAGE}" "${DEF_FILE}"

echo "[build] done. test import:"
apptainer exec "${APPTAINER_IMAGE}" python -c "import lerobot; print('lerobot OK')"
