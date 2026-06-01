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
- The log shows the loss as steps progress, e.g. (from the CPU smoke run, 2 steps):
  ```text
  step:1 smpl:2 ep:0 epch:0.02 loss:90.602 grdn:1423.650 lr:1.0e-05 ...
  step:2 smpl:4 ep:0 epch:0.04 loss:69.332 grdn:1264.084 lr:1.0e-05 ...
  End of training
  ```
  On Miyabi with `TRAIN_STEPS=2000` you see many more steps and the loss trends down.
- A run appears in the shared W&B (`WANDB_PROJECT`/`WANDB_ENTITY`) with a loss curve.
- A checkpoint is written under `OUTPUT_DIR/<job>/checkpoints/<step>/pretrained_model`.

## Try it locally first

`train.sh` is driven by environment variables, so you can rehearse on CPU with the
synthetic dataset (this is exactly what `make smoke` does):

```bash
python tools/make_synthetic_dataset.py --format lerobot --root .smoke/synthetic
DATA_REPO=handson/synthetic DATASET_ROOT=.smoke/synthetic OUTPUT_DIR=.smoke/outputs \
  POLICY_DEVICE=cpu TRAIN_STEPS=2 BATCH_SIZE=2 \
  CHUNK_SIZE=8 N_OBS_STEPS=1 N_ACTION_STEPS=8 PRETRAINED_BACKBONE_WEIGHTS=null \
  bash 03_train/train.sh
```

If something goes wrong, recall the four typical patterns in `challenges/debug/`
(OOM / offline / missing bind / wrong queue name).
