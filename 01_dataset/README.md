# 01_dataset — Inspect a dataset and the Hub (Notion Step 4)

## Goal

Get hands-on with the LeRobot data format (`LeRobotDataset`, v3.0) and feel what
"robot learning data" actually is — a collection of tensors. Load a dataset from the
Hub and check observation/action shapes, fps, and camera images with your own eyes.

## Prerequisites

- You have edited `config.env` and run `source config.env` (need `DATA_REPO`, `HF_HOME`).
- You have run `env/predownload_hf.sh` on the login node to pre-download `DATA_REPO`
  (so it can be read offline via `HF_HUB_OFFLINE=1`).

## What you use

- [`explore.ipynb`](./explore.ipynb) — loads `LeRobotDataset`, checks shapes, visualizes an image.

The notebook is meant to be opened in Jupyter / VS Code on the login node (or an
interactive node). No heavy training, so CPU is enough.

## Expected output (self-check cues)

Run the notebook top to bottom. You succeed if you can confirm all of the following:

- `dataset.meta.fps` (e.g. 30) plus `dataset.num_episodes`, `dataset.num_frames` are printed.
- `dataset[0].keys()` includes `action`, `observation.state`, `observation.images.*`.
- `dataset[0]['action'].shape` is `(action_dim,)`, and `observation.images.*` is a `(C, H, W)` tensor.
- One camera image is plotted and shows a robot's-eye view.
- Specifying `delta_timestamps` adds a leading time axis `T` to the same key's shape
  (e.g. `(T, C, H, W)`).

If it doesn't work:

- `HF_HUB_OFFLINE`-related errors → check that pre-download (`env/predownload_hf.sh`)
  is done and that `HF_HOME` points at the same shared area used during pre-download.
