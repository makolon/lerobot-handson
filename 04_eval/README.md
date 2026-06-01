# 04_eval — Evaluation (Notion Step 8)

## Goal

Evaluate a **distributed checkpoint** in the LIBERO simulation and produce a success
rate. The design lets you see "numbers of a robot doing the task" even if your own
training doesn't finish in time (it does not depend on training completion).

## Prerequisites

- `config.env` has been `source`d (need `CKPT_REPO`, `APPTAINER_IMAGE`, `HF_HOME`, `OUTPUT_DIR`).
- You pre-downloaded `CKPT_REPO` (the distributed checkpoint) with `env/predownload_hf.sh`.
- The container has the LIBERO eval dependencies (see the TODO in `env/apptainer.def`).

## Layout

| File | Role |
|------|------|
| [`eval.pbs`](./eval.pbs) | `#PBS` resource spec + calls `eval.sh` via `apptainer exec --nv` |
| [`eval.sh`](./eval.sh) | evaluates the distributed checkpoint in LIBERO via `lerobot-eval` |

## Run

```bash
source config.env
qsub 04_eval/eval.pbs
qstat
```

To evaluate a checkpoint you trained yourself, pass the `pretrained_model` path under
`OUTPUT_DIR` instead of `config.env`'s `CKPT_REPO` to `eval.sh` (see the in-script comment).

## Expected output (self-check cues)

- The log shows per-episode success/failure and a final **success rate** (e.g. `0.6`).
- Eval videos/rollouts are written under `OUTPUT_DIR` (depending on config).
- Eval metrics are recorded in the shared W&B (same project as the training run).

If something goes wrong:

- LIBERO import/environment errors → check that sim dependencies are in the container
  (the LIBERO TODO in `env/apptainer.def`).
