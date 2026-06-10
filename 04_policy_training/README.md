# 04_policy_training — Submit a training job (main / Notion Step 7)

## Goal

Train a LeRobot policy (default: ACT, because it is lightweight) as a PBS job on a
Miyabi GH200 compute node. Experience the full loop of "data → training → checkpoint".

VLAs (π0 / SmolVLA, etc.) are heavy and hard to fit in 105 minutes, so the **default
is a lightweight policy + few steps**. The design prioritizes the experience of
"it completes and a checkpoint is produced".

## Prerequisites

- `config.env` has been `source`d (sets `QUEUE_NAME`, `GROUP`, `APPTAINER_IMAGE`,
  `DATA_REPO`, `HF_HOME`, `OUTPUT_DIR`, `WANDB_*`, … and creates your `USER_DIR`).
- The shared image and dataset are already staged by the organizer (you build/download
  nothing — see [`MAINTAINER.md`](../MAINTAINER.md) §3). Your checkpoints land under your
  personal `OUTPUT_DIR=${USER_DIR}/outputs`, not your ~24 GB personal quota.

## Layout (the script is the source of truth)

| File | Role |
|------|------|
| [`train.pbs`](./train.pbs) | `#PBS` resource spec + wrapper that calls `train.sh` via `apptainer exec --nv` |
| [`train.sh`](./train.sh) | the `lerobot-train` body; arguments are assembled from `config.env` variables |

## Run

```bash
source config.env
qsub 04_policy_training/train.pbs        # submit the job
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
synthetic dataset:

```bash
python 03_dataset_conversion/make_synthetic_dataset.py --format lerobot --root .smoke/synthetic
DATA_REPO=handson/synthetic DATASET_ROOT=.smoke/synthetic OUTPUT_DIR=.smoke/outputs \
  POLICY_DEVICE=cpu TRAIN_STEPS=2 BATCH_SIZE=2 \
  CHUNK_SIZE=8 N_OBS_STEPS=1 N_ACTION_STEPS=8 PRETRAINED_BACKBONE_WEIGHTS=null \
  bash 04_policy_training/train.sh
```

If something goes wrong, the usual suspects are OOM (lower `BATCH_SIZE`), a missing
`HF_HUB_OFFLINE=1`/`HF_HOME`, a forgotten `--bind`, or a wrong queue name.

## Train ACT on the shared LIBERO dataset

A concrete end-to-end recipe: download the LeRobot LIBERO dataset
([`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero) —
all suites, dual-camera + 8-D state + 7-D action) into the **shared** area once, then
train ACT from it offline. Everyone reuses the one shared copy.

| File | Role |
|------|------|
| [`download_libero.sh`](./download_libero.sh) | **login node**: download the dataset into `/work/gw13/share/handson/libero`, warm-load it (caches any format conversion), and pre-cache the ACT ResNet18 ImageNet backbone into `/work/gw13/share/handson/torch` |
| [`train_libero.sh`](./train_libero.sh) | LIBERO config layer over `train.sh` — sets `--policy.type=act`, `--dataset.root=$LIBERO_ROOT`, sensible steps/batch |
| [`train_libero.pbs`](./train_libero.pbs) | PBS wrapper: binds the shared dataset (read-only) + backbone cache, runs `train_libero.sh` offline on a GH200 |

### 1. Download into the share (login node, has internet)

```bash
source config.env                       # need APPTAINER_IMAGE / APPTAINER_MODULE
bash 04_policy_training/download_libero.sh
# -> /work/gw13/share/handson/libero   (dataset; `du -sh` to check size)
# -> /work/gw13/share/handson/torch    (ResNet18 backbone cache)
```

Compute nodes are offline, so this **must** run on the login node. It pulls the
dataset and the backbone once; later jobs read both from the share.

### 2. Train ACT offline (GH200 compute node)

```bash
source config.env
qsub -q "$QUEUE_NAME" -W group_list="$GROUP" \
     -l select=1 -l walltime="$WALLTIME" \
     04_policy_training/train_libero.pbs
qstat          # Q -> R -> done
cat libero_act_train.o<jobid>  # read THIS job by id (the *.o glob also catches OLD runs)
```

Checkpoints land under `OUTPUT_DIR/libero_act/checkpoints/<step>/pretrained_model`,
and (if `WANDB_*` is configured) a loss curve appears in the shared W&B.

**Tuning** (all optional — Miyabi can't forward the submit env, so set these by adding
`export LIBERO_...=...` lines to `config.env`, which the job sources): `LIBERO_STEPS`
(default 50000), `LIBERO_BATCH` (default 32 — a GH200's ~96 GB can go higher),
`LIBERO_JOB_NAME`, `LIBERO_ROOT` (dataset path), `LIBERO_REPO`. To skip the ImageNet
backbone entirely (random init, fully offline-safe without step 1's pre-cache),
add `PRETRAINED_BACKBONE_WEIGHTS=null`.

> The eval module ([`05_policy_evaluation`](../05_policy_evaluation/)) defaults to
> `--env.type=libero --env.task=libero_object`, so an ACT policy trained here is the
> matching thing to evaluate. Note in-sim eval needs the LIBERO simulator
> (`MUJOCO_GL=egl`), whose aarch64 deps are still unconfirmed — training itself does
> not need the simulator.
