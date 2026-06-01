# 02_convert — Convert your own data to LeRobot format (Notion Step 5)

## Goal

Experience the flow of converting your own "raw data" (observation/action time series)
into the `LeRobotDataset` v3.0 format. You define `features` (dtype/shape per key),
`fps`, and `robot_type` yourself, then write it out in the order
`add_frame` → `save_episode` → `finalize`.

## Prerequisites

- `config.env` has been `source`d (need `HF_USER` if you push).
- This step uses dummy synthetic data, so no extra pre-download is needed.

## What you use

- [`convert_sample.py`](./convert_sample.py) — a minimal example that builds a small
  LeRobotDataset from synthetic data.
  - Default is local-save only. With `--push` it pushes to a repo under `HF_USER` (optional).

```bash
# Just build locally (recommended; works offline)
python 02_convert/convert_sample.py

# Push to the Hub (requires HF_USER, needs network, run on the login node)
python 02_convert/convert_sample.py --push
```

## Expected output (self-check cues)

- `LeRobotDataset.create(...)` completes without error and a dataset directory appears locally.
- A summary like "wrote N episodes / M frames" is printed to stdout.
- Immediately reloading with `LeRobotDataset(repo_id)` shows that `dataset[0]['action'].shape`
  matches the `features` you defined.
- With `--push`, a repo `${HF_USER}/<dataset-name>` is created on the Hub.

Key points:

- **Always call `finalize()`** (otherwise the parquet files are corrupt and won't load).
- Each entry in `features` has `{"dtype", "shape", "names"}`. See the in-script comments.
- To convert real-robot data, use this script's structure as a template and swap out
  the contents of `add_frame`.
