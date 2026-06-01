# lerobot-handson

105分ハンズオン「LeRobot を用いたロボット学習」の **実行コードの正本**（single source of truth）です。
データ確認 → LeRobot形式への変換 → 学習ジョブ投入 → 評価、の一巡を、スパコン
**Miyabi（JCAHPC / NVIDIA GH200 Grace Hopper, aarch64）** 上で体験します。

説明・進行・当日の生情報（キュー名・課金番号など）は Notion 側が担います。本リポジトリは
Notion から Step 番号・パスで参照されます。 <!-- TODO: Notionトップ URL -->

## pin している LeRobot バージョン

| 項目 | 値 |
|------|----|
| リリースタグ | `v0.5.1` |
| コミットハッシュ | `1396b9fab7aecddd10006c33c47a487ffdcb54b4` |
| 参照ドキュメント | https://huggingface.co/docs/lerobot/index |

> CLI（`lerobot-train` / `lerobot-eval` 等）の引数は、この pin した版の公式ドキュメント・
> `--help` を根拠にしています。確証が持てない引数には `# TODO(lerobot): v0.5.1 のドキュメントで要確認`
> コメントを付けています。

## 使い方（最短手順）

```bash
git clone <this-repo-url> lerobot-handson
cd lerobot-handson
cp config.env.example config.env
$EDITOR config.env      # 当日値（キュー名・課金番号・W&B・HF など）を埋める
source config.env       # 各スクリプトはこの変数を前提に動く
```

各スクリプトは冒頭で必要な環境変数が未設定なら**明確なエラーで停止**します（fail-fast）。
当日変わる値はスクリプトに直書きしていません。すべて `config.env` 経由で渡します。

## ディレクトリと Notion Step の対応

| ディレクトリ | 内容 | Notion Step |
|--------------|------|-------------|
| `slides/` | アーキテクチャ座学（Marp） | Step 3 |
| `01_dataset/` | データセットと Hub の確認 | Step 4 |
| `02_convert/` | LeRobot 形式への変換 | Step 5 |
| `env/` | Apptainer イメージ / HF 事前DL | Step 6（環境準備） |
| `03_train/` | 学習ジョブ（メイン） | Step 7 |
| `04_eval/` | 評価 | Step 8 |
| `challenges/debug/` | 本命1：壊れたジョブのデバッグ | （応用） |
| `challenges/leaderboard/` | 本命2：短時間チューニング | （応用） |
| `cheatsheet/` | qsub/qstat/qdel 早見表 | 全体 |

運営向けの運用方針（pin タグ・`step-XX-start` タグ・`solutions` ブランチ）は
[`MAINTAINER.md`](./MAINTAINER.md) を参照。

## 設計の前提（このハンズオン固有）

- **GH200 = aarch64**。計算ノードは**オフライン**。Python 依存は NGC の aarch64 PyTorch
  コンテナ（Apptainer）前提。HF 資産はログインノードで事前DLし、計算ジョブでは
  `HF_HUB_OFFLINE=1` と共有領域の `HF_HOME` を使う（[`env/`](./env/) 参照）。
- スケジューラは **PBS 系**（`qsub`/`qstat`/`qdel`）を仮定。**ただし Miyabi 固有の正確な
  仕様は未検証**（下記チェックリスト参照）。
- 全員が同じ read-only リポジトリを clone するだけ。成果共有は共有 W&B が担う。
- 遅れた人向けに各 Step 先頭に git タグ（`step-01-start` … `step-08-start`）を切れる設計。
  かつ各 Step のスクリプトは前 Step の結果に依存せず**自己完結**で動きます。

---

## ⚠️ 実施前チェックリスト（実環境で必ず確認）

作成者は GPU も Miyabi も持たず、実走確認をしていません。以下は **実施前に実環境で確認**
してください。該当箇所のスクリプトには `# TODO(miyabi)` / `# TODO(lerobot)` コメントがあります。

- [ ] **Miyabi のキュー名**（`QUEUE_NAME`）— GH200 計算ノード用 / 対話ノード用の正確な名称
- [ ] **課金グループの書式**（`GROUP`）と `#PBS` での指定フラグ（`-P` か `--group` か等）
- [ ] **`module` 名** — Apptainer/Singularity を load する `module load` の正確なモジュール名
- [ ] **PBS ディレクティブの書式** — ノード/GPU/walltime 指定（`-l select=...:ngpus=...` 等）が Miyabi で正しいか
- [ ] **Apptainer ビルドの実走** — `env/apptainer.def` が NGC aarch64 ベースで実際にビルドできるか、`lerobot[extras]` が aarch64 で解決するか
- [ ] **HF 事前DLの容量** — データセット/チェックポイントの実サイズと `HF_HOME` 共有領域の空き
- [ ] **`lerobot-train` / `lerobot-eval` の引数** — v0.5.1 の `--help` で arg 名（`--batch_size`/`--steps`/`--policy.device`/`--wandb.enable` 等）を確認
- [ ] **LIBERO シミュレーション依存** — 評価に必要な extras / 環境変数が計算ノードで揃うか
- [ ] **共有 W&B のオフライン挙動** — 計算ノードがオフラインの場合、`WANDB_MODE=offline` + 後段 sync が必要か
