# 03_train — 学習ジョブの投入（メイン / Notion Step 7）

## 目的

LeRobot のポリシー（既定: ACT。軽量なため）を、Miyabi の GH200 計算ノード上で
PBS ジョブとして学習する。「データ → 学習 → checkpoint」が回る一巡を体験する。

VLA（π0 / SmolVLA 等）は重く 105 分に収まりにくいので、**既定は軽量ポリシー +
少ステップ**にしている。「完走して checkpoint が出る」体験を最優先する設計。

## 前提

- `config.env` を `source` 済み（`QUEUE_NAME`, `GROUP`, `APPTAINER_IMAGE`, `DATA_REPO`,
  `HF_HOME`, `OUTPUT_DIR`, `WANDB_*` などが必要）。
- `env/build_image.sh` でイメージをビルド済み。
- `env/predownload_hf.sh` で `DATA_REPO` を共有 `HF_HOME` に事前DL済み
  （計算ノードはオフラインのため）。

## 構成（正本はスクリプト）

| ファイル | 役割 |
|----------|------|
| [`train.pbs`](./train.pbs) | `#PBS` 資源指定 + `apptainer exec --nv` で `train.sh` を呼ぶラッパ |
| [`train.sh`](./train.sh) | `lerobot-train` 本体。引数は `config.env` 由来の変数で組む |

## 実行

```bash
source config.env
qsub 03_train/train.pbs        # ジョブ投入
qstat                          # 状態確認（cheatsheet/ 参照）
```

ログは PBS の標準出力/エラー（`*.out` / `*.err`）に出る。

## 期待される出力（自己診断の手がかり）

- `qstat` でジョブが `Q`（待ち）→ `R`（実行）→ 完了、と遷移する。
- ログに `lerobot-train` の起動行と、step が進むごとの loss が出る。
- 共有 W&B（`WANDB_PROJECT`/`WANDB_ENTITY`）に run が現れ、loss 曲線が描かれる。
- `OUTPUT_DIR` 配下に checkpoint（`pretrained_model` 等）が生成される。

うまくいかない時は `challenges/debug/` の典型 4 パターン（OOM / オフライン /
bind 漏れ / キュー名誤り）を思い出す。
