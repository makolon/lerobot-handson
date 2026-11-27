# 05_policy_evaluation — Evaluation (Notion Step 8)

## Goal

Evaluate a checkpoint in the LIBERO simulation and produce a success rate — normally
**your own Section 4 checkpoint** (`POLICY_PATH`), or a distributed checkpoint
(`CKPT_REPO`) if the organizer staged one, so you still get numbers when your own
training did not finish.

## Prerequisites

- `config.env` has been `source`d (sets `CKPT_REPO`, `APPTAINER_IMAGE`, `HF_HOME`, `OUTPUT_DIR`).
- `CKPT_REPO` is empty by default. If the organizer set it and staged it with
  `env/predownload_hf.sh` (see [`MAINTAINER.md`](../MAINTAINER.md) §3), `eval.sh` evaluates it
  when no `POLICY_PATH` is given. Eval output goes to your personal `OUTPUT_DIR`.
- The container has the LIBERO eval dependencies (the `lerobot[libero]` extra in `env/apptainer.def`).

## Layout

| File | Role |
|------|------|
| [`eval.pbs`](./eval.pbs) | `#PBS` resource spec + calls `eval.sh` via `apptainer exec --nv` |
| [`eval.sh`](./eval.sh) | evaluates `POLICY_PATH` (or `CKPT_REPO`) in LIBERO via `lerobot-eval` |

## Run

```bash
source config.env
# your own checkpoint (forward POLICY_PATH with -v; Miyabi rejects -V)
POLICY_PATH=$OUTPUT_DIR/act_base/checkpoints/last/pretrained_model \
qsub -q "$QUEUE_NAME" -W group_list="$GROUP" -l select=1 -l walltime="$WALLTIME" \
     -v POLICY_PATH 05_policy_evaluation/eval.pbs
qstat
```

Interactively (inside `apptainer shell --nv --bind "$SHARED_DIR" ...`):
`bash 05_policy_evaluation/eval.sh --policy.path=$OUTPUT_DIR/act_base/checkpoints/last/pretrained_model`.
Results go to `$OUTPUT_DIR/eval/<run name>` (here `$OUTPUT_DIR/eval/act_base`).

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
       --output_dir=.smoke/outputs/eval/smoke_train --policy.device=cpu
[eval] DRY_RUN=1 -> command construction verified, not executing.
```

Note `eval.batch_size` must be ≤ `eval.n_episodes` (lerobot raises otherwise); `eval.sh`
defaults the batch size to the episode count.

## Expected output (self-check cues, real run)

- LIBERO runs **every task in the suite** (10 for `libero_object`), each for
  `EVAL_EPISODES` episodes, then prints `Overall Aggregated Metrics` with `pc_success`
  (a percentage, e.g. `60.0`) and `n_episodes` (e.g. 100).
- Rollout videos (up to 10 per task) are written under
  `$OUTPUT_DIR/eval/<run>/videos/<suite>_<task_id>/eval_episode_*.mp4` — `lerobot-eval` always
  records these. `eval_info.json` holds `per_task` / `per_group` / `overall` results.
- `lerobot-eval` does not log to W&B; read the numbers from the log or `eval_info.json`.

If something goes wrong:

- LIBERO import/environment errors → check that sim dependencies are in the container
  (the `lerobot[libero]` extra in `env/apptainer.def`).
