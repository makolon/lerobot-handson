# 05_policy_evaluation — Evaluation (Notion Step 8)

## Goal

Evaluate a **distributed checkpoint** in the LIBERO simulation and produce a success
rate. The design lets you see "numbers of a robot doing the task" even if your own
training doesn't finish in time (it does not depend on training completion).

## Prerequisites

- `config.env` has been `source`d (sets `CKPT_REPO`, `APPTAINER_IMAGE`, `HF_HOME`, `OUTPUT_DIR`).
- The distributed checkpoint (`CKPT_REPO`) is already in the shared `HF_HOME`, staged by
  the organizer (see [`MAINTAINER.md`](../MAINTAINER.md) §3). Eval output goes to your
  personal `OUTPUT_DIR`.
- The container has the LIBERO eval dependencies (the `lerobot[libero]` extra in `env/apptainer.def`).

## Layout

| File | Role |
|------|------|
| [`eval.pbs`](./eval.pbs) | `#PBS` resource spec + calls `eval.sh` via `apptainer exec --nv` |
| [`eval.sh`](./eval.sh) | evaluates the distributed checkpoint in LIBERO via `lerobot-eval` |

## Run

```bash
source config.env
qsub 05_policy_evaluation/eval.pbs
qstat
```

To evaluate a checkpoint you trained yourself, pass the `pretrained_model` path under
`OUTPUT_DIR` instead of `config.env`'s `CKPT_REPO` to `eval.sh` (see the in-script comment).

## Verify the command without LIBERO

LIBERO needs simulation deps that may be absent off-Miyabi, so `eval.sh` supports
`DRY_RUN=1` to print the exact command without running it:

```bash
DRY_RUN=1 POLICY_PATH=.smoke/outputs/smoke_train/checkpoints/last/pretrained_model \
  OUTPUT_DIR=.smoke/outputs bash 05_policy_evaluation/eval.sh
```
```text
[eval] lerobot-eval --policy.path=.../pretrained_model --env.type=libero \
       --env.task=libero_object --eval.n_episodes=10 --eval.batch_size=10 \
       --output_dir=.smoke/outputs/eval_pretrained_model --policy.device=cpu
[eval] DRY_RUN=1 -> command construction verified, not executing.
```

Note `eval.batch_size` must be ≤ `eval.n_episodes` (lerobot raises otherwise); `eval.sh`
defaults the batch size to the episode count.

## Expected output (self-check cues, real run)

- The log shows per-episode success/failure and a final **success rate** (e.g. `0.6`).
- Eval videos/rollouts are written under `OUTPUT_DIR` (depending on config).
- Eval metrics are recorded in the shared W&B (same project as the training run).

If something goes wrong:

- LIBERO import/environment errors → check that sim dependencies are in the container
  (the `lerobot[libero]` extra in `env/apptainer.def`).
