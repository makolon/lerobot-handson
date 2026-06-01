# 03_train — Submit a training job (main / Notion Step 7)

## Goal

Train a LeRobot policy (default: ACT, because it is lightweight) as a PBS job on a
Miyabi GH200 compute node. Experience the full loop of "data → training → checkpoint".

VLAs (π0 / SmolVLA, etc.) are heavy and hard to fit in 105 minutes, so the **default
is a lightweight policy + few steps**. The design prioritizes the experience of
"it completes and a checkpoint is produced".

## Prerequisites

- `config.env` has been `source`d (need `QUEUE_NAME`, `GROUP`, `APPTAINER_IMAGE`,
  `DATA_REPO`, `HF_HOME`, `OUTPUT_DIR`, `WANDB_*`, etc.).
- You built the image with `env/build_image.sh`.
- You pre-downloaded `DATA_REPO` into the shared `HF_HOME` with `env/predownload_hf.sh`
  (because compute nodes are offline).

## Layout (the script is the source of truth)

| File | Role |
|------|------|
| [`train.pbs`](./train.pbs) | `#PBS` resource spec + wrapper that calls `train.sh` via `apptainer exec --nv` |
| [`train.sh`](./train.sh) | the `lerobot-train` body; arguments are assembled from `config.env` variables |

## Run

```bash
source config.env
qsub 03_train/train.pbs        # submit the job
qstat                          # check status (see cheatsheet/)
```

Logs go to the PBS stdout/stderr (`*.out` / `*.err`).

## Expected output (self-check cues)

- `qstat` shows the job moving `Q` (queued) → `R` (running) → done.
- The log shows the `lerobot-train` startup line and the loss as steps progress.
- A run appears in the shared W&B (`WANDB_PROJECT`/`WANDB_ENTITY`) with a loss curve.
- A checkpoint (e.g. `pretrained_model`) is generated under `OUTPUT_DIR`.

If something goes wrong, recall the four typical patterns in `challenges/debug/`
(OOM / offline / missing bind / wrong queue name).
