#!/usr/bin/env bash
# =============================================================================
# build_image.sh  —  build the Apptainer image on the login node
# -----------------------------------------------------------------------------
# Run on: the Miyabi [login node] (not a compute node).
#
# This is the ORGANIZER's one-time staging build (see MAINTAINER.md §3): it runs
# ONCE into the shared area, and every participant just READS the resulting .sif.
# Participants do NOT need to build. The guard below stops anyone from clobbering
# the shared image by accident — point BUILD_OUTPUT at your own area to experiment.
#
# Verified on the Miyabi-G (GH200, aarch64) login node 2026-06-08: a `--fakeroot`
# build runs to completion on the login node (~15 GB SIF). subuid/subgid are
# configured and newuidmap/newgidmap carry the needed caps, so NO `--remote` /
# dedicated build host is required. The one catch: APPTAINER_TMPDIR/CACHEDIR MUST
# live on local /tmp (NOT the setgid /work tree), or fakeroot's gid mapping fails
# — enforced below. The build is CPU/network heavy, so run it once, off-peak;
# it is not something the whole class does at the same time.
# =============================================================================
set -euo pipefail

: "${APPTAINER_IMAGE:?source config.env and set APPTAINER_IMAGE}"

# --- Output destination -----------------------------------------------------
# Default to the SHARED image (APPTAINER_IMAGE) so the organizer's one-time
# staging build writes there. Anyone who just wants to *experience* a build
# should send it to their OWN area instead, leaving the shared .sif untouched:
#   BUILD_OUTPUT="${USER_DIR}/images/lerobot.sif" bash env/build_image.sh
: "${BUILD_OUTPUT:=${APPTAINER_IMAGE}}"

# --- Safety: never silently clobber the shared image ------------------------
# If the target already exists, refuse unless it is clearly intentional. This
# stops someone who runs this script by mistake from breaking the .sif the whole
# class reads.
if [[ -e "${BUILD_OUTPUT}" ]]; then
  if [[ ! -w "${BUILD_OUTPUT}" ]]; then
    echo "[build] ERROR: ${BUILD_OUTPUT} already exists and is not writable by you." >&2
    echo "        That is the SHARED image — you do not need to (re)build it." >&2
    echo "        To experience a build, point BUILD_OUTPUT at your own area:" >&2
    echo "          BUILD_OUTPUT=\"\${USER_DIR}/images/lerobot.sif\" bash env/build_image.sh" >&2
    exit 1
  fi
  if [[ "${BUILD_OUTPUT}" == "${APPTAINER_IMAGE}" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[build] ERROR: the shared image already exists:" >&2
    echo "          ${BUILD_OUTPUT}" >&2
    echo "        Refusing to overwrite it. To rebuild the shared image on purpose," >&2
    echo "        re-run with FORCE_REBUILD=1; to experiment, set BUILD_OUTPUT to a" >&2
    echo "        path under your own USER_DIR." >&2
    exit 1
  fi
fi

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
echo "[build] sif      : ${BUILD_OUTPUT}"
echo "[build] tmpdir   : ${APPTAINER_TMPDIR}"
echo "[build] cachedir : ${APPTAINER_CACHEDIR}"

mkdir -p "$(dirname "${BUILD_OUTPUT}")"

apptainer build --fakeroot "${BUILD_OUTPUT}" "${DEF_FILE}"

echo "[build] done. test import:"
apptainer exec "${BUILD_OUTPUT}" python -c "import lerobot; print('lerobot OK')"
