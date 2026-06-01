# Cheatsheet (source for the PDF later)

> Miyabi-specific values are `<TODO>`. The scheduler is **assumed PBS-family**
> (confirm in the official manual).
> TODO(miyabi): confirm the qsub/qstat/qdel option syntax, queue names, billing flags.

## PBS job operations

| What you want | Command |
|---------------|---------|
| Submit a job | `qsub 03_train/train.pbs` |
| Pass resources at submit | `qsub -q "$QUEUE_NAME" -P "$GROUP" -l select=1:ngpus=1 -l walltime="$WALLTIME" -v ALL 03_train/train.pbs` |
| List your jobs | `qstat -u $USER` |
| All jobs | `qstat` |
| Job details | `qstat -f <jobid>` |
| Delete a job | `qdel <jobid>` |
| Reserve interactive node | `qsub -I -q "$QUEUE_NAME_INTERACTIVE" ...` (TODO(miyabi): confirm syntax) |

### Job status symbols (general PBS)
- `Q` = queued / `R` = running / `E` = exiting / `C` or gone = completed
- `H` = held (e.g. waiting on a dependency)

## Reading logs
- stdout/stderr go to `<JobName>.o<jobid>` / `.e<jobid>` (merged with `#PBS -j oe`)
- Read the **tail** first (`tail -n 50 <logfile>`). Errors are often at the end.

## Key environment variables

| Variable | Role |
|----------|------|
| `HF_HOME` | HF cache location. Points at the **shared area** (match it to pre-download) |
| `HF_HUB_OFFLINE=1` | Don't let compute nodes fetch from HF (required offline) |
| `WANDB_PROJECT` / `WANDB_ENTITY` | Shared W&B destination (result sharing) |
| `WANDB_MODE=offline` | When no network. `wandb sync` later |
| `APPTAINER_IMAGE` | Absolute path of the built `.sif` |

## Apptainer

| What you want | Command |
|---------------|---------|
| Run with GPU | `apptainer exec --nv "$APPTAINER_IMAGE" <cmd>` |
| Expose an area | `--bind /path:/path` (don't forget data/output/HF_HOME) |
| Pass env vars | `--env HF_HUB_OFFLINE=1 --env HF_HOME=$HF_HOME` |
| Enter a shell | `apptainer shell --nv "$APPTAINER_IMAGE"` |

## LeRobot typical commands (v0.5.1 / the script is the source of truth)

> Policy: full command text lives in scripts. This is a quick "which one to invoke".

| Purpose | Entry point | Note |
|---------|-------------|------|
| Train | `qsub 03_train/train.pbs` | body is `03_train/train.sh` (`lerobot-train`) |
| Eval | `qsub 04_eval/eval.pbs` | body is `04_eval/eval.sh` (`lerobot-eval`) |
| Tuning | `bash challenges/leaderboard/run_tuning.sh` | assumes an interactive node |
| Inspect data | `01_dataset/explore.ipynb` | `LeRobotDataset` |
| Convert | `python 02_convert/convert_sample.py` | `--push` to the Hub |
| Pre-download | `bash env/predownload_hf.sh` | on the login node |

## Common failures → what to suspect first (maps to challenges/debug)
- `CUDA out of memory` → lower `--batch_size`
- Hangs on external connection → missing `HF_HUB_OFFLINE=1` / `HF_HOME`
- `FileNotFoundError` → is the data area in `apptainer --bind`?
- `qsub` rejected / killed immediately → check queue name / walltime
