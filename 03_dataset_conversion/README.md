# 03_dataset_conversion — Convert your own data to LeRobot format (Notion Step 5)

## Goal

Experience the flow of converting your own "raw data" (observation/action time series)
into the `LeRobotDataset` v3.0 format. You define `features` (dtype/shape per key),
`fps`, and `robot_type` yourself, then write it out in the order
`add_frame` → `save_episode` → `finalize`.

## Prerequisites

- `config.env` has been `source`d (need `HF_USER` only if you `--push`) and the
  Apptainer module is loaded (`module load "$APPTAINER_MODULE"`).
- `HF_HOME` exists on the host. The reload step makes `datasets` write a cache under
  it, and apptainer's `--bind` needs the source path to exist first:
  ```bash
  mkdir -p "$HF_HOME"
  ```
- Raw episodes to convert. Generate synthetic ones (no net/GPU), inside the container:
  ```bash
  apptainer exec "$APPTAINER_IMAGE" \
    python 03_dataset_conversion/make_synthetic_dataset.py --format raw --out .smoke/raw
  ```

## What you use

- [`convert_sample.py`](./convert_sample.py) — reads the raw `.npz` episodes, defines
  `features` / `fps` / `robot_type`, runs `create → add_frame → save_episode → finalize`,
  then **reloads and asserts** the shapes round-trip.
  - Default is local save. `--push` pushes to a repo under `HF_USER` (optional, needs net).

Run it inside the container. The `--bind`/`--env HF_HOME` flags mirror what
`04_policy_training/train.pbs` and `05_policy_evaluation/eval.pbs` do: only `$PWD` is auto-mounted, so
`$HF_HOME` (outside `$PWD`) must be bound explicitly or the reload step fails with
`Read-only file system: .../hf_home`.

```bash
# Build locally (offline)
apptainer exec \
  --bind "$HF_HOME:$HF_HOME" --env "HF_HOME=$HF_HOME" \
  "$APPTAINER_IMAGE" \
  python 03_dataset_conversion/convert_sample.py --raw .smoke/raw --root .smoke/converted

# Push to the Hub (requires HF_USER, needs network, run on the login node)
apptainer exec \
  --bind "$HF_HOME:$HF_HOME" --env "HF_HOME=$HF_HOME" \
  "$APPTAINER_IMAGE" \
  python 03_dataset_conversion/convert_sample.py --raw .smoke/raw --root .smoke/converted --push
```

## Expected output (self-check cues)

```text
[convert] wrote 4 episodes / 96 frames -> .smoke/converted (repo_id=local-user/handson-convert-sample)
[convert] reload OK: episodes=4 frames=96 action(6,) image(3, 64, 64)
```

The `reload OK` line means the asserts passed: a freshly created dataset reloads and its
`action` / `image` shapes match the `features` you defined. With `--push`, a repo
`${HF_USER}/handson-convert-sample` is created on the Hub.

Key points:

- **Always call `finalize()`** (otherwise the parquet files are corrupt and won't load).
- Each entry in `features` has `{"dtype", "shape", "names"}`. See the in-script comments.
- To convert real-robot data, use this script's structure as a template and swap out
  the contents of `add_frame`.
