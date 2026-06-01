#!/usr/bin/env bash
# =============================================================================
# run_tuning.sh  —  短時間チューニング対決のランナー（本命2）
# -----------------------------------------------------------------------------
# 【対話ノード前提】: バッチ投入ではなく、確保済みの対話/デバッグノードで直接実行する。
#   （例: 対話ノードを確保 → コンテナに入る or apptainer exec → 本スクリプト）
#   TODO(miyabi): 対話ノードの確保方法（qsub -I 等）は公式マニュアルで要確認。
#
# 公平性のため固定: step 数 / walltime / データセット / ポリシー種別。
# いじれるノブ（引数 or 環境変数）:
#   CHUNK_SIZE : action horizon（一度に予測する行動ステップ数）
#   LR         : learning rate
#   BATCH_SIZE : バッチサイズ
#   OBS_STEPS  : 観測の時間窓（observation steps）
#   AUG        : 画像augmentation on/off
#
# TODO(lerobot): 各ノブに対応する lerobot-train の config キー名は v0.5.1 の
#   `lerobot-train --help` で要確認。下の綴りは ACT 系の一般的な名称に基づく推定。
#     chunk size  -> --policy.chunk_size （+ --policy.n_action_steps 連動の場合あり）
#     obs steps   -> --policy.n_obs_steps
#     lr          -> --optimizer.lr もしくは --policy.optimizer_lr
#     aug         -> 画像変換の有効化フラグ（dataset.image_transforms.enable 等）
# =============================================================================
set -euo pipefail

# --- 固定値（公平性のため変更しない）---
FIXED_STEPS="${FIXED_STEPS:-1000}"
POLICY_TYPE="${POLICY_TYPE:-act}"

# --- fail-fast: 共有 W&B が無いとリーダーボードにならない ---
: "${DATA_REPO:?DATA_REPO 未設定 (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR 未設定}"
: "${WANDB_PROJECT:?WANDB_PROJECT 未設定: 共有W&Bに出さないと順位が見えません}"

# --- ノブ（既定値つき）---
CHUNK_SIZE="${CHUNK_SIZE:-50}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OBS_STEPS="${OBS_STEPS:-1}"
AUG="${AUG:-off}"

# run 名にノブを埋めて W&B 上で区別しやすくする
RUN_NAME="lb_${POLICY_TYPE}_cs${CHUNK_SIZE}_lr${LR}_bs${BATCH_SIZE}_obs${OBS_STEPS}_aug${AUG}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# augmentation フラグの組み立て
AUG_ARGS=()
if [[ "${AUG}" == "on" ]]; then
  # TODO(lerobot): 画像aug有効化キーを v0.5.1 で要確認
  AUG_ARGS+=( "--dataset.image_transforms.enable=true" )
fi

WANDB_ARGS=( "--wandb.enable=true" "--wandb.project=${WANDB_PROJECT}" )
[[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "<"* ]] && \
  WANDB_ARGS+=( "--wandb.entity=${WANDB_ENTITY}" )

echo "[tuning] ${RUN_NAME}"

# 対話ノードで既にコンテナ内にいるならそのまま lerobot-train を呼ぶ。
# コンテナ外から回すなら apptainer exec --nv "${APPTAINER_IMAGE}" で本体を包む。
lerobot-train \
  --policy.type="${POLICY_TYPE}" \
  --dataset.repo_id="${DATA_REPO}" \
  --steps="${FIXED_STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_obs_steps="${OBS_STEPS}" \
  --optimizer.lr="${LR}" \
  --output_dir="${OUTPUT_DIR}/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --policy.device=cuda \
  "${AUG_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

echo "[tuning] done. 共有 W&B (${WANDB_PROJECT}) で ${RUN_NAME} のスコアを確認。"
