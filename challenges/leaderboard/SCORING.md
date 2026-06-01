# SCORING — How to read scores and rankings

## Metrics

Look at two metrics. They play different roles, so keep both in mind.

### 1. validation loss (during-training metric)
- The validation loss reported during training. **Lower is better.**
- Use it to gauge "training efficiency" — does it drop fast, does it plateau low?
- In a short competition, this is the most straightforward axis of comparison.

### 2. eval success rate (evaluation metric — the real one)
- The fraction of tasks succeeded in LIBERO (`0.0`–`1.0`). **Higher is better.**
- "Low loss ≠ actually works" can happen. Weight this more for the final verdict.
- For how to evaluate, see [`../../04_eval/`](../../04_eval/) (evaluate the same checkpoint).

> In a short slot you may not get to run eval. In that case a practical scheme is:
> val loss as the primary metric, eval success rate as the deciding metric (for those
> who managed to run it).

## How to read the ranking in the shared W&B

Since everyone writes to the same `WANDB_PROJECT` / `WANDB_ENTITY`, the W&B Workspace
is the leaderboard.

1. Open that project in the browser (`https://wandb.ai/<entity>/<project>`).
2. In the **Runs table**, add columns for `val/loss` (min) and `eval/success_rate` (max).
3. Sort by that column to get the ranking. The run names embed the knobs
   (e.g. `lb_act_cs50_lr1e-4_bs8_obs1_augon`), so you can see at a glance which setting worked.
4. In **Charts**, overlay loss curves from multiple runs to compare convergence
   speed and stability.

## Hints (which knobs matter)

- chunk size (action horizon): larger is smoother but tends to make training harder.
- learning rate: too large diverges, too small won't drop enough in a short time.
- batch size: larger is more stable but watch for OOM (see `challenges/debug/broken_01`).
- observation steps: more history → more expressiveness but more compute.
- image aug: curbs overfitting, but too strong is counterproductive.

> The knob → `lerobot-train` flag mapping (`--policy.chunk_size`, `--policy.n_obs_steps`,
> `--policy.optimizer_lr`, `--batch_size`, `--dataset.image_transforms.enable`) is
> implemented in `run_tuning.sh` and verified against lerobot 0.5.1. Note that for ACT
> `n_action_steps` must be ≤ `chunk_size`, so the tuner ties them together.
