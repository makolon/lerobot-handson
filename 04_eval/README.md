# 04_eval — 評価（Notion Step 8）

## 目的

**配布済みチェックポイント**を LIBERO シミュレーション上で評価し、success rate を出す。
自分の学習が時間内に終わらなくても、ここで「ロボットがタスクをこなす数字」を見て
達成感を得られる設計（学習完走に依存しない）。

## 前提

- `config.env` を `source` 済み（`CKPT_REPO`, `APPTAINER_IMAGE`, `HF_HOME`, `OUTPUT_DIR` が必要）。
- `env/predownload_hf.sh` で `CKPT_REPO`（配布済みチェックポイント）を事前DL済み。
- コンテナに LIBERO 評価依存が入っていること（`env/apptainer.def` の TODO 参照）。

## 構成

| ファイル | 役割 |
|----------|------|
| [`eval.pbs`](./eval.pbs) | `#PBS` 資源指定 + `apptainer exec --nv` で `eval.sh` を呼ぶ |
| [`eval.sh`](./eval.sh) | `lerobot-eval` で配布済み checkpoint を LIBERO 評価 |

## 実行

```bash
source config.env
qsub 04_eval/eval.pbs
qstat
```

自分で学習した checkpoint を評価したい場合は、`config.env` の `CKPT_REPO` の代わりに
`OUTPUT_DIR` 配下の `pretrained_model` パスを `eval.sh` に渡す（スクリプト内コメント参照）。

## 期待される出力（自己診断の手がかり）

- ログに各エピソードの成否と、最終的な **success rate**（例 `0.6` 等）が出る。
- 評価動画/ロールアウトが `OUTPUT_DIR` 配下に出力される（設定による）。
- 共有 W&B に eval の指標が記録される（学習 run と同じ project）。

うまくいかない時:

- LIBERO 関連の import/環境エラー → コンテナに sim 依存が入っているか
  （`env/apptainer.def` の LIBERO TODO）を確認。
