# チートシート（後で PDF 化する元）

> Miyabi 固有値は `<TODO>`。スケジューラは **PBS系を仮定**（公式マニュアルで要確認）。
> TODO(miyabi): qsub/qstat/qdel のオプション書式・キュー名・課金フラグを要確認。

## PBS ジョブ操作

| やりたいこと | コマンド |
|--------------|----------|
| ジョブ投入 | `qsub 03_train/train.pbs` |
| 投入時に資源を渡す | `qsub -q "$QUEUE_NAME" -P "$GROUP" -l select=1:ngpus=1 -l walltime="$WALLTIME" -v ALL 03_train/train.pbs` |
| 自分のジョブ一覧 | `qstat -u $USER` |
| 全ジョブ | `qstat` |
| ジョブ詳細 | `qstat -f <jobid>` |
| ジョブ削除 | `qdel <jobid>` |
| 対話ノード確保 | `qsub -I -q "$QUEUE_NAME_INTERACTIVE" ...`（TODO(miyabi): 書式要確認） |

### ジョブ状態の記号（PBS 一般）
- `Q` = 待ち / `R` = 実行中 / `E` = 終了処理中 / `C` or 消える = 完了
- `H` = ホールド（依存待ちなど）

## ログの見方
- 標準出力/エラーは `<JobName>.o<jobid>` / `.e<jobid>`（`#PBS -j oe` で統合）
- まず**末尾**を読む（`tail -n 50 <logfile>`）。エラーは最後に出ることが多い。

## 主要な環境変数

| 変数 | 役割 |
|------|------|
| `HF_HOME` | HF キャッシュ置き場。**共有領域**を指す（事前DLと一致させる） |
| `HF_HUB_OFFLINE=1` | 計算ノードで HF へ取りに行かせない（オフライン必須） |
| `WANDB_PROJECT` / `WANDB_ENTITY` | 共有 W&B の宛先（成果共有） |
| `WANDB_MODE=offline` | ネット不可時。後で `wandb sync` |
| `APPTAINER_IMAGE` | ビルド済み `.sif` の絶対パス |

## Apptainer

| やりたいこと | コマンド |
|--------------|----------|
| GPU 付きで実行 | `apptainer exec --nv "$APPTAINER_IMAGE" <cmd>` |
| 領域を見せる | `--bind /path:/path`（データ/出力/HF_HOME を忘れず） |
| 環境変数を渡す | `--env HF_HUB_OFFLINE=1 --env HF_HOME=$HF_HOME` |
| シェルに入る | `apptainer shell --nv "$APPTAINER_IMAGE"` |

## LeRobot 典型コマンド（v0.5.1 / 正本はスクリプト）

> コマンド全文はスクリプトに置く方針。ここは「どれを叩くか」の早見。

| 目的 | 入口 | 備考 |
|------|------|------|
| 学習 | `qsub 03_train/train.pbs` | 本体は `03_train/train.sh`（`lerobot-train`） |
| 評価 | `qsub 04_eval/eval.pbs` | 本体は `04_eval/eval.sh`（`lerobot-eval`） |
| チューニング | `bash challenges/leaderboard/run_tuning.sh` | 対話ノード前提 |
| データ確認 | `01_dataset/explore.ipynb` | `LeRobotDataset` |
| 変換 | `python 02_convert/convert_sample.py` | `--push` で Hub へ |
| 事前DL | `bash env/predownload_hf.sh` | ログインノードで |

## よくある失敗 → まず疑う所（challenges/debug 対応）
- `CUDA out of memory` → `--batch_size` を下げる
- 外部接続で固まる → `HF_HUB_OFFLINE=1` / `HF_HOME` の設定漏れ
- `FileNotFoundError` → `apptainer --bind` にデータ領域が入っているか
- `qsub` が弾かれる / すぐ kill → キュー名 / walltime を確認
