#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh  —  run the whole exercise path offline, on CPU, no Miyabi
# -----------------------------------------------------------------------------
# Pipeline:
#   1. generate a synthetic LeRobotDataset (+ raw episodes)
#   2. load it and print shapes
#   3. convert the raw episodes -> LeRobotDataset (with reload asserts)
#   4. train ACT for a few CPU steps via the real 03_train/train.sh
#   5. run the leaderboard tuner for a few CPU steps via run_tuning.sh
#   6. verify 04_eval/eval.sh command construction (DRY_RUN)
#   7. (optional) execute 01_dataset/explore.ipynb if jupyter is available
#
# GPU / network / LIBERO are NOT required. The ImageNet backbone download is
# disabled (PRETRAINED_BACKBONE_WEIGHTS=null).
#
# Usage:
#   bash tools/smoke_test.sh
#   PYTHON=/path/to/venv/bin/python PATH=/path/to/venv/bin:$PATH bash tools/smoke_test.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
SMOKE_DIR="${SMOKE_DIR:-.smoke}"
DS_ROOT="${SMOKE_DIR}/synthetic"
RAW_DIR="${SMOKE_DIR}/raw"
CONVERTED="${SMOKE_DIR}/converted"
OUT_DIR="${SMOKE_DIR}/outputs"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
rm -rf "${SMOKE_DIR}"
mkdir -p "${OUT_DIR}"

hr() { printf '\n===== %s =====\n' "$1"; }

hr "1/7 generate synthetic LeRobotDataset"
"${PYTHON}" tools/make_synthetic_dataset.py --format lerobot --root "${DS_ROOT}" \
  --episodes 4 --length 24 | grep -v 'Map:'

hr "1b   generate raw episodes (for convert)"
"${PYTHON}" tools/make_synthetic_dataset.py --format raw --out "${RAW_DIR}" --episodes 4 --length 24

hr "2/7 load the dataset and print shapes"
"${PYTHON}" - "$DS_ROOT" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("handson/synthetic", root=sys.argv[1])
s = ds[0]
print(f"loaded: episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.meta.fps}")
print(f"keys={sorted(s.keys())}")
print(f"action={tuple(s['action'].shape)} state={tuple(s['observation.state'].shape)} "
      f"image={tuple(s['observation.images.front'].shape)}")
PY

hr "3/7 convert raw -> LeRobotDataset (with reload asserts)"
"${PYTHON}" 02_convert/convert_sample.py --raw "${RAW_DIR}" --root "${CONVERTED}" | grep -v 'Map:'

hr "4/7 train ACT a few CPU steps via 03_train/train.sh"
DATA_REPO=handson/synthetic DATASET_ROOT="${DS_ROOT}" OUTPUT_DIR="${OUT_DIR}" \
  POLICY_TYPE=act POLICY_DEVICE=cpu TRAIN_STEPS=2 BATCH_SIZE=2 \
  CHUNK_SIZE=8 N_OBS_STEPS=1 N_ACTION_STEPS=8 \
  PRETRAINED_BACKBONE_WEIGHTS=null SAVE_FREQ=2 LOG_FREQ=1 JOB_NAME=smoke_train \
  bash 03_train/train.sh 2>&1 | grep -iE 'lerobot-train|step:|End of training|checkpoints ->'

hr "5/7 leaderboard tuner a few CPU steps via run_tuning.sh"
DATA_REPO=handson/synthetic DATASET_ROOT="${DS_ROOT}" OUTPUT_DIR="${OUT_DIR}" \
  POLICY_DEVICE=cpu FIXED_STEPS=2 BATCH_SIZE=2 CHUNK_SIZE=8 OBS_STEPS=1 LR=1e-4 AUG=on \
  PRETRAINED_BACKBONE_WEIGHTS=null \
  bash challenges/leaderboard/run_tuning.sh 2>&1 | grep -iE 'WARNING|step:|End of training|done\.'

hr "6/7 verify eval.sh command construction (DRY_RUN)"
DRY_RUN=1 POLICY_PATH="${OUT_DIR}/smoke_train/checkpoints/last/pretrained_model" \
  OUTPUT_DIR="${OUT_DIR}" EVAL_EPISODES=10 POLICY_DEVICE=cpu \
  bash 04_eval/eval.sh

hr "7/7 execute explore.ipynb (if jupyter available)"
if "${PYTHON}" -c "import nbconvert" 2>/dev/null; then
  # nbconvert runs the kernel in the notebook's dir, so pass an ABSOLUTE dataset root.
  DATA_REPO=handson/synthetic LEROBOT_DATASET_ROOT="${ROOT}/${DS_ROOT}" \
    "${PYTHON}" -m nbconvert --to notebook --execute --output-dir "${ROOT}/${SMOKE_DIR}" \
    --output explore.executed.ipynb 01_dataset/explore.ipynb >/dev/null 2>&1 \
    && echo "notebook executed OK -> ${SMOKE_DIR}/explore.executed.ipynb" \
    || echo "notebook execution FAILED (see nbconvert error)"
else
  echo "nbconvert not installed -> skipping notebook execution (cells still run in Jupyter)"
fi

hr "SMOKE TEST PASSED"
echo "checkpoint: $(find "${OUT_DIR}/smoke_train/checkpoints" -name pretrained_model -type d | head -1)"
