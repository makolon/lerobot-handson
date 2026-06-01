# 01_dataset — データセットと Hub の確認（Notion Step 4）

## 目的

LeRobot のデータ形式（`LeRobotDataset`, v3.0）に触れ、「ロボットの学習データとは
何のテンソルの集まりなのか」を体感する。Hub 上のデータセットを読み込み、
observation/action の shape・fps・カメラ画像を自分の目で確認する。

## 前提

- `config.env` を編集して `source config.env` 済み（`DATA_REPO`, `HF_HOME` が必要）。
- ログインノードで `env/predownload_hf.sh` を実行し、`DATA_REPO` を事前DL済み
  （オフラインでも `HF_HUB_OFFLINE=1` で読めるようにするため）。

## 使うもの

- [`explore.ipynb`](./explore.ipynb) — `LeRobotDataset` を読み込み、shape 確認・画像可視化。

ノートブックはログインノード（または対話ノード）の Jupyter / VS Code で開く想定。
重い学習はしないので CPU で十分。

## 期待される出力（自己診断の手がかり）

ノートブックを上から実行して、以下がすべて確認できれば成功:

- `dataset.meta.fps`（例: 30 など）と `dataset.num_episodes`, `dataset.num_frames` が表示される。
- `dataset[0].keys()` に `action`, `observation.state`, `observation.images.*` が含まれる。
- `dataset[0]['action'].shape` が `(action_dim,)`、`observation.images.*` が `(C, H, W)` のテンソル。
- カメラ画像が 1 枚プロットされ、ロボット視点の絵が見える。
- `delta_timestamps` を指定すると同じキーの shape 先頭に時間軸 `T` が増える
  （例 `(T, C, H, W)`）ことが確認できる。

うまくいかない時:

- `HF_HUB_OFFLINE` 関連エラー → 事前DL（`env/predownload_hf.sh`）が済んでいるか、
  `HF_HOME` が事前DL時と同じ共有領域を指しているか確認。
