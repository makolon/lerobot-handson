# 02_convert — 自前データを LeRobot 形式へ変換（Notion Step 5）

## 目的

手元の「生データ（observation/action の時系列）」を `LeRobotDataset` v3.0 形式に
変換する流れを体験する。`features`（各キーの dtype/shape）・`fps`・`robot_type` を
自分で定義し、`add_frame` → `save_episode` → `finalize` の順で書き出す。

## 前提

- `config.env` を `source` 済み（push する場合は `HF_USER` が必要）。
- ここではダミーの合成データを使うので、追加の事前DLは不要。

## 使うもの

- [`convert_sample.py`](./convert_sample.py) — 合成データから小さな LeRobotDataset を作る最小例。
  - 既定はローカル保存のみ。`--push` を付けると `HF_USER` のリポジトリへ push する（任意）。

```bash
# ローカルに作るだけ（推奨・オフラインでも可）
python 02_convert/convert_sample.py

# Hub に push する場合（HF_USER 必須・ネット必要・ログインノードで）
python 02_convert/convert_sample.py --push
```

## 期待される出力（自己診断の手がかり）

- `LeRobotDataset.create(...)` がエラーなく完了し、ローカルに dataset ディレクトリができる。
- 標準出力に「`N episodes / M frames を書き出しました`」のような要約が出る。
- 直後に `LeRobotDataset(repo_id)` で読み直し、`dataset[0]['action'].shape` が
  自分で定義した `features` と一致することを確認できる。
- `--push` 時は Hub 上に `${HF_USER}/<dataset名>` が作成される。

ポイント:

- **`finalize()` を必ず呼ぶ**（呼ばないと parquet が壊れて読み込めない）。
- `features` の各エントリは `{"dtype", "shape", "names"}` を持つ。詳細はスクリプト内コメント参照。
- 実機データの変換は本スクリプトの構造をテンプレに、`add_frame` の中身を差し替える。
