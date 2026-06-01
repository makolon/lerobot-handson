#!/usr/bin/env bash
# =============================================================================
# eval.sh  —  lerobot-eval 本体（コンテナ内で実行される）
# -----------------------------------------------------------------------------
# 設計意図:
#   既定では【配布済みチェックポイント(CKPT_REPO)】を LIBERO で評価する。
#   こうすることで、自分の学習が時間内に終わらなくても success rate を見て
#   達成感を得られる（学習完走に依存しない）。
#
# 確認できた最小例（公式 README より）:
#   lerobot-eval --policy.path=lerobot/pi0_libero_finetuned \
#                --env.type=libero --env.task=libero_object --eval.n_episodes=10
#
# TODO(lerobot): --env.task の選択肢、--eval.batch_size、--output_dir、
#                --policy.device の正確な綴りは v0.5.1 の `lerobot-eval --help` で要確認。
# =============================================================================
set -euo pipefail

# --- fail-fast ---
: "${CKPT_REPO:?CKPT_REPO 未設定 (config.env)}"
: "${OUTPUT_DIR:?OUTPUT_DIR 未設定}"
: "${EVAL_EPISODES:=10}"
: "${ENV_TASK:=libero_object}"   # TODO(lerobot): 配布 ckpt に合う task 名に要調整

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# 評価対象の checkpoint。既定は配布済み repo。
# 自分の学習成果を評価したい場合は下を OUTPUT_DIR 配下のパスに差し替える:
#   POLICY_PATH="${OUTPUT_DIR}/${JOB_NAME}/checkpoints/last/pretrained_model"
POLICY_PATH="${CKPT_REPO}"

echo "[eval] policy=${POLICY_PATH} env=libero task=${ENV_TASK} episodes=${EVAL_EPISODES}"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --env.type=libero \
  --env.task="${ENV_TASK}" \
  --eval.n_episodes="${EVAL_EPISODES}" \
  --output_dir="${OUTPUT_DIR}/eval_$(basename "${POLICY_PATH}")" \
  --policy.device=cuda

echo "[eval] done. success rate と動画は出力ディレクトリ/ログを参照。"
