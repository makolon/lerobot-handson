# MAINTAINER — 運営向け運用ガイド

このリポジトリは「実行コードの正本」。**毎回の開催で更新するのは原則 `config.env`
（各参加者が編集）と Notion 側だけ**で済むように設計してある。コード自体は安定運用する。

## 1. LeRobot の pin タグ運用

- `README.md` に **タグ名（`v0.5.1`）とコミットハッシュ**を明記している。
- `env/apptainer.def` の `From:` / インストール行・`%labels` も同じ版に合わせる。
- バージョンを上げる時の手順:
  1. 新タグの `lerobot-train` / `lerobot-eval` の `--help` で引数変更を確認。
  2. `*.sh` 内の `# TODO(lerobot)` 箇所を実機で検証して更新。
  3. `README.md` のタグ/ハッシュ表、`apptainer.def` の `%labels` を更新。
  4. イメージを再ビルドし、最低限 import と `--help` が通ることを確認。

## 2. `step-XX-start` タグ（遅れた人の救済）

各 Step の開始地点に git タグを切れる設計（救済の保険その1）。

```bash
# 例: 各 Step の "開始時点" の main にタグを打つ
git tag step-01-start <commit>
git tag step-02-start <commit>
# ... step-08-start まで
git push origin --tags
```

- 遅れた参加者は `git checkout step-05-start` でそこから追いつける。
- **保険その2**: 各 Step のスクリプトは前 Step の実行結果に依存せず**自己完結**で動く
  （データもHFから取得、出力先も独立）。タグが無くても追いつける二重構成。
- Step とディレクトリの対応は `README.md` の対応表を参照。

## 3. `solutions` ブランチ（本命1の答え）

壊れたジョブ（`challenges/debug/broken_*.pbs`）の **修正版**と解説
（`challenges/debug/SOLUTIONS.md`）は **`solutions` ブランチにのみ**置く。
**`main` には答えを混ぜない**。

```bash
# 作成（初回）
git switch -c solutions
# challenges/debug/ に修正版 + SOLUTIONS.md を置いてコミット
git push -u origin solutions

# 当日: main を配布。詰まった人には solutions ブランチ or Notion トグルを案内。
```

- `main` の `challenges/debug/README.md` は「答えは Notion トグル or solutions ブランチ」
  とだけ案内する（解答は書かない）。
- LeRobot 版を上げた時は `main` の `broken_*.pbs` と `solutions` の修正版を**両方**揃える。

## 4. 毎回の開催でやること（最小）

- [ ] Notion の当日情報（キュー名・課金番号・W&B project/entity・データ repo）を更新。
- [ ] 参加者に `config.env.example` → `config.env` 編集を案内（リポジトリは read-only）。
- [ ] ログインノードでイメージビルド（`env/build_image.sh`）と HF 事前DL
      （`env/predownload_hf.sh`）が共有領域で済んでいるか確認。
- [ ] 必要なら `step-XX-start` タグを今回の HEAD に打ち直す。

## 5. 検証できていない点（引き継ぎ）

`README.md` 末尾の「実施前チェックリスト」と各スクリプトの `# TODO(miyabi)` /
`# TODO(lerobot)` を参照。**捏造で TODO を消さない**こと。Miyabi 実機で確認でき次第、
該当 TODO を実値に置き換え、チェックリストを縮める。
