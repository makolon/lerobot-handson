#!/usr/bin/env bash
# =============================================================================
# train.sh  —  lerobot-train 本体（コンテナ内で実行される）
# -----------------------------------------------------------------------------
# 呼び出し元: 03_train/train.pbs（apptainer exec 経由）。
# 単体テストもできる: コンテナ内で `bash 03_train/train.sh`。
#
# フォールバック設計:
#   既定は「軽量ポリシー(ACT) + 少ステップ(TRAIN_STEPS)」で、105分内に必ず
#   "流れる"（loss が動き checkpoint が出る）体験を担保する。本格学習は
#   TRAIN_STEPS を増やす / より重いポリシーに変える。
#
# TODO(lerobot): lerobot-train の引数名は v0.5.1 の `lerobot-train --help` で要確認。
#   確認できた最小例（公式 README より）:
#     lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
#   下記の --batch_size / --steps / --output_dir / --policy.device / --wandb.* は
#   draccus 由来の config キーだが、正確な綴りは要確認。
# =============================================================================
set -euo pipefail

# --- fail-fast: 必須変数 ---
: "${DATA_REPO:?DATA_REPO 未設定 (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR 未設定}"
: "${POLICY_TYPE:=act}"
: "${TRAIN_STEPS:=2000}"
: "${BATCH_SIZE:=8}"
: "${JOB_NAME:=handson_${POLICY_TYPE}}"

# W&B（共有）。未設定でも動くよう緩めにする。
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" && "${WANDB_PROJECT}" != "<"* ]]; then
  # TODO(lerobot): wandb の有効化フラグ名（--wandb.enable / --wandb.project /
  #                --wandb.entity）を v0.5.1 で要確認。
  WANDB_ARGS+=( "--wandb.enable=true" "--wandb.project=${WANDB_PROJECT}" )
  [[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "<"* ]] && \
    WANDB_ARGS+=( "--wandb.entity=${WANDB_ENTITY}" )
fi

echo "[train] policy=${POLICY_TYPE} dataset=${DATA_REPO} steps=${TRAIN_STEPS} batch=${BATCH_SIZE}"
echo "[train] output=${OUTPUT_DIR}/${JOB_NAME}"

# 計算ノードはオフライン。train.pbs 側で export 済みだが単体実行向けに再確認。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

lerobot-train \
  --policy.type="${POLICY_TYPE}" \
  --dataset.repo_id="${DATA_REPO}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${TRAIN_STEPS}" \
  --output_dir="${OUTPUT_DIR}/${JOB_NAME}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  "${WANDB_ARGS[@]}"

# --- フォールバック案（短時間で確実に "流す" 別解）---
# 既存の配布済みチェックポイントから短時間 fine-tune したい場合は、上の
# --policy.type の代わりに事前学習済みポリシーを起点にする:
#   TODO(lerobot): pretrained から始める引数（例 --policy.path=${CKPT_REPO}）の
#                  正確な指定方法を v0.5.1 で要確認。
echo "[train] done. checkpoints -> ${OUTPUT_DIR}/${JOB_NAME}"
