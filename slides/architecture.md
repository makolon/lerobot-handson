---
marp: true
title: Architecture lecture for robot-learning models
paginate: true
---

# Architectures of robot-learning models

An overview of the main policies you handle in LeRobot

- Intuition for Action Chunking
- ACT / Diffusion Policy
- π0 / π0.5 (VLA + flow matching)
- GR00T N1.5 (cross-embodiment)
- SmolVLA (small VLA)

> Figures are skeletons (ASCII / bullet points). Fill in verbally on the day.

---

# First: the imitation-learning frame

Learn observation → action.

```
   obs o_t                        action a_t
 ┌─────────────┐   policy π     ┌──────────────┐
 │ camera image│ ───────────▶  │ joint command│
 │ joint state │                │ (e.g. 7-dof) │
 └─────────────┘                └──────────────┘
```

- Training data = human demonstrations (observation/action time series)
- Issue: predicting one step at a time accumulates error (compounding error)

---

# Intuition for Action Chunking

Instead of one step, **predict "the next H steps of actions" all at once**.

```
 t        t+1   t+2   ...   t+H
 │ observe
 └─▶ predict [a_t, a_t+1, ..., a_t+H]   <- chunk (action horizon)
```

- Benefit: fewer high-frequency re-decisions; smoother behavior, robust to error accumulation
- Knob: **chunk size = action horizon** (larger = smoother / harder to train)

---

# ACT (Action Chunking Transformer)

- Predicts an **action chunk** with a Transformer + CVAE
- Input: multi-camera images + joint state; output: the next H steps of actions
- Relatively **lightweight and fast to train** → the default policy for this hands-on

```
 [imgs, state] ─▶ Encoder(Transformer) ─▶ z ─▶ Decoder ─▶ [a_t ... a_t+H]
                                     (CVAE latent)
```

---

# Diffusion Policy

- Generates actions with a **diffusion model (denoising)**
- Starts from noise and gradually denoises into an action sequence, conditioned on the observation

```
 noise ──denoise×K──▶ action chunk
            ▲
      condition: obs o_t
```

- Good at representing multimodality (multiple valid behaviors)
- Inference is iterative (K steps) and somewhat heavy

---

# π0 / π0.5 (VLA + flow matching)

- **VLA = Vision-Language-Action**: action generation on top of a VLM
- Language instruction + image → action. Generates continuous actions via **flow matching**
- π0.5 is an enhanced version oriented toward generalization / open-world

```
 [image + "put the cup on the plate"]
        │  VLM backbone
        ▼
   flow matching head ─▶ continuous action chunk
```

- Highly expressive but **heavy** → not the default for this hands-on

---

# GR00T N1.5 (cross-embodiment)

- NVIDIA's foundation model for humanoids
- **cross-embodiment**: learn/transfer across different robot morphologies
- Aims for broad generalization with large-scale data + simulation

```
 diverse embodiments (arm/hand/humanoid) ─┐
 diverse tasks & data                     ├─▶ shared policy backbone
 sim + real                               ─┘
```

---

# SmolVLA (small VLA)

- A **small, practical VLA** from the LeRobot community
- Keeps the design of large VLAs while running on **realistic compute budgets**
- Suits education / edge / small-data fine-tuning

> A bridge for "I want to try a VLA, but π0 is still too heavy."

---

# Summary: axes for choosing

| policy | weight | characteristics |
|--------|--------|-----------------|
| ACT | light | action chunk + Transformer/CVAE. fast |
| Diffusion Policy | medium | strong at multimodality. iterative inference |
| SmolVLA | medium | small VLA. language-conditioned |
| π0 / π0.5 | heavy | VLA + flow matching. highly expressive |
| GR00T N1.5 | heavy | cross-embodiment foundation |

The default for this hands-on is **ACT** (prioritizing a "it flows" experience in 105 min).
