# tools — offline scaffolding for the exercises

These let the whole exercise path run on a laptop (CPU, no network, no Miyabi), so the
repo can be sanity-checked before the event and participants can rehearse.

| File | What it does |
|------|--------------|
| [`make_synthetic_dataset.py`](./make_synthetic_dataset.py) | Generate dummy robot data. `--format lerobot` writes a ready-to-load `LeRobotDataset`; `--format raw` writes per-episode `.npz` for `02_convert` to convert. |
| [`smoke_test.sh`](./smoke_test.sh) | Run the full path: generate → load → convert → train ACT (few CPU steps via `03_train/train.sh`) → tune (`run_tuning.sh`) → verify `eval.sh` command → execute `01_dataset/explore.ipynb`. |

## Run

Requires an environment with `lerobot==0.5.1` (and `matplotlib`/`nbconvert` for the
notebook step). From the repo root:

```bash
make smoke
# or point at a specific interpreter (put its bin on PATH so lerobot-train resolves):
PATH=/path/to/venv/bin:$PATH PYTHON=/path/to/venv/bin/python make smoke
```

Scratch output goes to `.smoke/` (git-ignored). `make clean` removes it.

## Notes

- The smoke test sets `PRETRAINED_BACKBONE_WEIGHTS=null` so ACT does not download
  ImageNet weights — it is fully offline once `lerobot` is installed.
- Images use the `image` feature dtype (PNG frames), so no video codec is required.
- This is **bucket 2** (runs anywhere). Miyabi-specific values stay in `config.env` /
  `# TODO(miyabi)` placeholders and are not exercised here.
