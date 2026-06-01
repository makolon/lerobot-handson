# challenges/leaderboard — 短時間チューニング対決（本命2）

## これは何か

固定の小データ・固定 step 数・固定 walltime の中で、**ノブ（ハイパラ）を調整して
できるだけ良いスコアを出す**ミニ競技です。共有 W&B 上に全員の run が並ぶので、
そこが実質のリーダーボードになります。

**対話ノード前提**（バッチ投入ではなく、確保した対話/デバッグノードで直接回す想定）。
短いイテレーションで「回して→W&Bを見て→ノブを変える」をくり返します。

## いじれるノブ

`run_tuning.sh` に引数 or 環境変数で渡します（詳細はスクリプト冒頭コメント）:

| ノブ | 変数 | 意味 |
|------|------|------|
| chunk size (action horizon) | `CHUNK_SIZE` | 一度に予測する行動ステップ数 |
| learning rate | `LR` | 学習率 |
| batch size | `BATCH_SIZE` | バッチサイズ |
| observation steps | `OBS_STEPS` | 観測の時間窓 |
| 画像 augmentation | `AUG` | `on` / `off` |

固定されるもの（公平性のため変えない）: step 数・walltime・データセット・ポリシー種別。

## 実行

```bash
source config.env
# 例: chunk size と lr を変えて回す
CHUNK_SIZE=50 LR=1e-4 AUG=on bash challenges/leaderboard/run_tuning.sh
```

## スコアの見方

[`SCORING.md`](./SCORING.md) を参照（指標と W&B 上の順位の読み方）。
