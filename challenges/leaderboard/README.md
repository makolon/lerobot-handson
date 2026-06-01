# challenges/leaderboard — Short tuning competition (Bonus 2)

## What this is

A mini-competition where, within a fixed small dataset, fixed step count, and fixed
walltime, you **tune the knobs (hyperparameters) to get the best score you can**. Since
everyone's runs line up in the shared W&B, that effectively becomes the leaderboard.

**Assumes an interactive node** (you run it directly on a reserved interactive/debug
node rather than via batch submission). You iterate quickly: "run → look at W&B →
change a knob".

## Knobs you can turn

Pass them to `run_tuning.sh` via arguments or environment variables (see the script's
header comment):

| Knob | Variable | Meaning |
|------|----------|---------|
| chunk size (action horizon) | `CHUNK_SIZE` | number of action steps predicted at once |
| learning rate | `LR` | learning rate |
| batch size | `BATCH_SIZE` | batch size |
| observation steps | `OBS_STEPS` | observation time window |
| image augmentation | `AUG` | `on` / `off` |

Fixed (for fairness; do not change): step count, walltime, dataset, policy type.

## Run

```bash
source config.env
# Example: run with a different chunk size and lr
CHUNK_SIZE=50 LR=1e-4 AUG=on bash challenges/leaderboard/run_tuning.sh
```

## Reading the score

See [`SCORING.md`](./SCORING.md) (the metrics and how to read the ranking in W&B).
