# Cheatsheet (source for the PDF later)

> Miyabi-G (GH200) is **PBS Pro**, verified live 2026-06-08. Billing project goes via
> `-W group_list=<GROUP>` (Miyabi rejects PBS Pro's `-P`). One node = one GH200; the GPU
> is auto-allocated by `-l select=1` (no `:ngpus=`). List queues with `qstat --rsc`.

## PBS job operations

| What you want | Command |
|---------------|---------|
| Submit a job | `qsub 04_policy_training/train.pbs` |
| Pass resources at submit | `qsub -q "$QUEUE_NAME" -W group_list="$GROUP" -l select=1 -l walltime="$WALLTIME" 04_policy_training/train.pbs` (no -v/-V: Miyabi rejects forwarding the submit env; the job sources config.env) |
| List your jobs | `qstat` (Miyabi qstat has no -u flag) |
| Queues & your allocation | `qstat --rsc` / `qstat --limit` |
| Job details | `qstat -f <jobid>` |
| Delete a job | `qdel <jobid>` |
| Reserve interactive node | `qsub -I -q "$QUEUE_NAME_INTERACTIVE" -W group_list="$GROUP" -l select=1 -l walltime=00:15:00` |

### Job status symbols (general PBS)
- `Q` = queued / `R` = running / `E` = exiting / `C` or gone = completed
- `H` = held (e.g. waiting on a dependency)

## Reading logs
- stdout/stderr go to `<JobName>.o<jobid>` / `.e<jobid>` (merged with `#PBS -j oe`)
- Read the **tail** first (`tail -n 50 <logfile>`). Errors are often at the end.

## Key environment variables

| Variable | Role |
|----------|------|
| `HF_HOME` | HF cache location. `config.env` points it at the **shared area** (read-only for participants) |
| `USER_HF_HOME` | Your own writable HF cache — `export HF_HOME=$USER_HF_HOME` before downloading something yourself |
| `LIBERO_ROOT` | The shared LIBERO dataset folder — pass `--dataset.root=$LIBERO_ROOT` |
| `HF_HUB_OFFLINE=1` | Optional: read only from the cache (compute nodes have internet; don't set it when loading a dataset by repo_id) |
| `WANDB_PROJECT` / `WANDB_ENTITY` | Shared W&B destination (result sharing) |
| `WANDB_MODE=offline` / `--wandb.mode=offline` | No API key in the job. Upload later: `wandb sync $OUTPUT_DIR/<run>/wandb/offline-run-*` |
| `APPTAINER_IMAGE` | Absolute path of the built `.sif` |

## Apptainer

| What you want | Command |
|---------------|---------|
| Run with GPU | `apptainer exec --nv "$APPTAINER_IMAGE" <cmd>` |
| Expose an area | `--bind "$SHARED_DIR"` (only `$PWD`/`$HOME` are mounted automatically) |
| Pass env vars | `--env HF_HOME=$HF_HOME` (host env is passed through by default) |
| Enter a shell | `apptainer shell --nv --bind "$SHARED_DIR" "$APPTAINER_IMAGE"` |

## LeRobot typical commands (v0.5.1 / the script is the source of truth)

> Policy: full command text lives in scripts. This is a quick "which one to invoke".

| Purpose | Entry point | Note |
|---------|-------------|------|
| Train | `qsub 04_policy_training/train.pbs` | body is `04_policy_training/train.sh` (`lerobot-train`) |
| Eval | `qsub 05_policy_evaluation/eval.pbs` | body is `05_policy_evaluation/eval.sh` (`lerobot-eval`) |
| Inspect data | `02_imitation_learning/explore.ipynb` | `LeRobotDataset` |
| Convert | `python 03_dataset_conversion/convert_sample.py` | `--push` to the Hub |
| Pre-stage (organizer) | `bash 04_policy_training/download_libero.sh` / `bash env/predownload_hf.sh` | on the login node |

## Common failures → what to suspect first
- `CUDA out of memory` → lower `--batch_size`
- Tries to download the dataset / `Read-only file system` → missing `--dataset.root=$LIBERO_ROOT`,
  or you are writing to the shared `HF_HOME` (use `$USER_HF_HOME`)
- `FileNotFoundError` → is the data area in `apptainer --bind`? (`--bind "$SHARED_DIR"`)
- `qsub` rejected / killed immediately → check queue name / walltime / `-W group_list=` (not `-P`)
