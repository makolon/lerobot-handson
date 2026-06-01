---
marp: true
title: ロボット学習モデルのアーキテクチャ座学
paginate: true
---

# ロボット学習モデルのアーキテクチャ

LeRobot で扱う主要な方策（policy）を俯瞰する

- Action Chunking の直感
- ACT / Diffusion Policy
- π0 / π0.5（VLA + flow matching）
- GR00T N1.5（cross-embodiment）
- SmolVLA（小型 VLA）

> 図は骨子（ASCII / 箇条書き）。当日は口頭で補足。

---

# まず：模倣学習の枠組み

観測 → 行動 を学習する。

```
   観測 o_t                       行動 a_t
 ┌─────────────┐   policy π     ┌──────────┐
 │ camera 画像 │ ───────────▶  │ 関節指令 │
 │ 関節状態    │                │ (7dofなど)│
 └─────────────┘                └──────────┘
```

- 教師データ = 人間のデモ（observation/action の時系列）
- 課題: 1 ステップずつ予測すると誤差が累積（compounding error）

---

# Action Chunking の直感

1 ステップではなく **「これからの H ステップ分の行動」をまとめて予測**する。

```
 t        t+1   t+2   ...   t+H
 │ observe
 └─▶ predict [a_t, a_t+1, ..., a_t+H]   ← chunk (action horizon)
```

- 利点: 高頻度の再決定を減らし、滑らかで誤差累積に強い挙動
- ノブ: **chunk size = action horizon**（大きいほど滑らか／学習は難化）

---

# ACT (Action Chunking Transformer)

- Transformer + CVAE で **行動チャンク**を予測
- 入力: 複数カメラ画像 + 関節状態、出力: 次の H ステップの行動列
- 比較的**軽量で学習が速い** → 本ハンズオンの既定ポリシー

```
 [imgs, state] ─▶ Encoder(Transformer) ─▶ z ─▶ Decoder ─▶ [a_t ... a_t+H]
                                     (CVAE latent)
```

---

# Diffusion Policy

- 行動を **拡散モデル（denoising）** で生成
- ノイズから出発し、観測を条件に行動列へと徐々にデノイズ

```
 noise ──denoise×K──▶ action chunk
            ▲
      条件: 観測 o_t
```

- 多峰性（複数の正解挙動）を表現しやすい
- 推論は反復的（K ステップ）でやや重い

---

# π0 / π0.5（VLA + flow matching）

- **VLA = Vision-Language-Action**：VLM の上に行動生成を載せる
- 言語指示 + 画像 → 行動。**flow matching** で連続行動を生成
- π0.5 は汎化・オープンワールド指向の強化版

```
 [画像 + "put the cup on the plate"]
        │  VLM backbone
        ▼
   flow matching head ─▶ 連続行動チャンク
```

- 表現力が高い反面**重い** → 本ハンズオンでは既定にしない

---

# GR00T N1.5（cross-embodiment）

- NVIDIA の humanoid 向け基盤モデル
- **cross-embodiment**：異なるロボット形態をまたいで学習・転移
- 大規模データ + シミュレーションで広い汎化を狙う

```
 多様な embodiment(腕/ハンド/humanoid) ─┐
 多様なタスク・データ                  ├─▶ 共有 policy backbone
 sim + real                            ─┘
```

---

# SmolVLA（小型 VLA）

- LeRobot コミュニティ発の**小型・実用志向 VLA**
- 大型 VLA の設計を引き継ぎつつ、**現実的な計算資源**で動かす
- 教育・エッジ・少データ fine-tune に向く

> 「VLA を体験したいが π0 はまだ重い」層の橋渡し的存在。

---

# まとめ：使い分けの軸

| policy | 重さ | 特徴 |
|--------|------|------|
| ACT | 軽 | 行動チャンク + Transformer/CVAE。速い |
| Diffusion Policy | 中 | 多峰性に強い。推論反復 |
| SmolVLA | 中 | 小型 VLA。言語条件 |
| π0 / π0.5 | 重 | VLA + flow matching。高表現力 |
| GR00T N1.5 | 重 | cross-embodiment 基盤 |

本ハンズオンの既定は **ACT**（105 分で「流れる」体験を優先）。
