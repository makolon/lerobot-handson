# 02_imitation_learning — Inspect a dataset and the Hub (Notion Step 4)

## Goal

Get hands-on with the LeRobot data format (`LeRobotDataset`, v3.0) and feel what
"robot learning data" actually is — a collection of tensors. Load a dataset from the
Hub and check observation/action shapes, fps, and camera images with your own eyes.

## Prerequisites

- On Miyabi: `source config.env` (need `DATA_REPO`, `LIBERO_ROOT`). The organizer staged
  `DATA_REPO` once as a plain folder at `$LIBERO_ROOT`; the notebook reads it from there
  (LeRobot reads a local root or its own cache, not the generic Hub cache). Inside the
  container, bind the share: `apptainer exec --bind "$SHARED_DIR" ...`.
- Offline/standalone (laptop): build the synthetic dataset and point the notebook at it:
  ```bash
  python 03_dataset_conversion/make_synthetic_dataset.py --format lerobot --root .smoke/synthetic
  export DATA_REPO=handson/synthetic
  export LEROBOT_DATASET_ROOT=$PWD/.smoke/synthetic
  ```

## What you use

- [`explore.ipynb`](./explore.ipynb) — loads `LeRobotDataset`, prints metadata/shapes,
  takes a `delta_timestamps` window, runs one DataLoader batch, plots a frame and an
  episode's signals, and ends with three **🔧 Try it** exercises.

Open it in Jupyter / VS Code (login or interactive node). No GPU needed.

## Expected output (self-check cues)

Run the notebook top to bottom. Against the synthetic dataset you should see, e.g.:

```text
fps          = 10
num_episodes = 4
num_frames   = 96

=== features ===
observation.state               dtype=float32  shape=(6,)
action                          dtype=float32  shape=(6,)
observation.images.front        dtype=image    shape=(64, 64, 3)

action shape = (6,)
state  shape = (6,)
state shape without window : (6,)
state shape with    window : (3, 6)   # leading T=3
batch action shape : (4, 6)
batch image  shape : (4, 3, 64, 64)
episode 0 length: 24 frames
```

With a real Hub dataset the numbers differ (e.g. `fps=30`, larger images), but the
shape relationships are the same.

If it doesn't work:

- It tries to download / `Read-only file system` → the dataset was loaded without a local
  root. Check `LIBERO_ROOT` is set (or `LEROBOT_DATASET_ROOT`) and that the container was
  started with `--bind "$SHARED_DIR"` (only `$PWD`/`$HOME` are mounted automatically).
