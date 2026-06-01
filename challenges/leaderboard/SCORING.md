# SCORING — スコアと順位の見方

## 指標

2 つの指標で見ます。役割が違うので両方を意識してください。

### 1. validation loss（学習中の指標）
- 学習中に出る検証損失。**低いほど良い**。
- 速く下がるか・下げ止まりが低いかで「学習の効率」を測る。
- 短時間勝負なので、まずはここが素直な比較軸になる。

### 2. eval success rate（評価指標・本命）
- LIBERO 上でタスクを成功させた割合（`0.0`〜`1.0`）。**高いほど良い**。
- 「loss が低い ≠ 実際に動く」ことがある。最終的な勝敗はこちらを重視。
- 評価のやり方は [`../../04_eval/`](../../04_eval/) を参照（同じ checkpoint を評価する）。

> 短時間枠では eval まで回せないこともある。その場合は val loss を一次指標、
> eval success rate を（回せた人の）決勝指標とする運用が現実的。

## 共有 W&B 上での順位の読み方

全員が同じ `WANDB_PROJECT` / `WANDB_ENTITY` に書き込むので、W&B の Workspace が
そのままリーダーボードになります。

1. ブラウザで該当 project を開く（`https://wandb.ai/<entity>/<project>`）。
2. **Runs テーブル**で、列に `val/loss`（最小）や `eval/success_rate`（最大）を追加。
3. その列でソートすれば順位が出る。run 名にノブが埋まっている
   （例 `lb_act_cs50_lr1e-4_bs8_obs1_augon`）ので、どの設定が効いたか一目で分かる。
4. **Charts** で複数 run の loss 曲線を重ねると、収束の速さ・安定性を比較できる。

## ヒント（どのノブが効くか）

- chunk size（action horizon）: 大きいと滑らかだが学習は難しくなりがち。
- learning rate: 大きすぎると発散、小さすぎると短時間で下がりきらない。
- batch size: 大きいほど安定だが OOM 注意（`challenges/debug/broken_01` 参照）。
- observation steps: 履歴を増やすと表現力↑だが計算↑。
- 画像 aug: 過学習を抑えるが、強すぎると逆効果。

> 注意: 上記ノブ名に対応する `lerobot-train` の config キーは v0.5.1 で要確認
> （`run_tuning.sh` の `TODO(lerobot)` 参照）。
