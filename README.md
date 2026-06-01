# lerobot-handson

The **single source of truth for the runnable code** of the 105-minute hands-on
"Training robot-learning models with LeRobot". You go through the full loop —
inspect data → convert to LeRobot format → submit a training job → evaluate — on the
supercomputer **Miyabi (JCAHPC / NVIDIA GH200 Grace Hopper, aarch64)**.

Explanation, pacing, and day-of live information (queue names, billing numbers, etc.)
live on the Notion side. This repository is referenced from Notion by Step number and
path. <!-- TODO: Notion top URL -->

## Pinned LeRobot version

| Item | Value |
|------|-------|
| Release tag | `v0.5.1` |
| Commit hash | `1396b9fab7aecddd10006c33c47a487ffdcb54b4` |
| Reference docs | https://huggingface.co/docs/lerobot/index |

> The CLI arguments (`lerobot-train` / `lerobot-eval`, etc.) are based on the official
> docs and `--help` of this pinned version. Arguments we could not fully confirm are
> marked with `# TODO(lerobot): confirm against the v0.5.1 docs`.

## Quick start

```bash
git clone <this-repo-url> lerobot-handson
cd lerobot-handson
cp config.env.example config.env
$EDITOR config.env      # fill in day-of values (queue, billing group, W&B, HF, ...)
source config.env       # every script assumes these variables are set
```

Each script **fails fast with a clear error** if a required environment variable is
unset. Day-of values are never hard-coded in scripts; they all flow through `config.env`.

## Directory ↔ Notion Step mapping

| Directory | Contents | Notion Step |
|-----------|----------|-------------|
| `slides/` | Architecture lecture (Marp) | Step 3 |
| `01_dataset/` | Inspect a dataset and the Hub | Step 4 |
| `02_convert/` | Convert to LeRobot format | Step 5 |
| `env/` | Apptainer image / HF pre-download | Step 6 (env setup) |
| `03_train/` | Training job (main) | Step 7 |
| `04_eval/` | Evaluation | Step 8 |
| `challenges/debug/` | Bonus 1: debug a broken job | (advanced) |
| `challenges/leaderboard/` | Bonus 2: short tuning competition | (advanced) |
| `cheatsheet/` | qsub/qstat/qdel quick reference | (whole event) |

Operational policy for maintainers (pin tags, `step-XX-start` tags, the `solutions`
branch) is in [`MAINTAINER.md`](./MAINTAINER.md).

## Design assumptions (specific to this hands-on)

- **GH200 = aarch64**. Compute nodes are **offline**. Python dependencies assume the
  NGC aarch64 PyTorch container (Apptainer). HF assets are pre-downloaded on the login
  node, and compute jobs use `HF_HUB_OFFLINE=1` with an `HF_HOME` pointing at shared
  storage (see [`env/`](./env/)).
- The scheduler is assumed to be **PBS-family** (`qsub`/`qstat`/`qdel`). **However the
  exact Miyabi-specific specs are unverified** (see the checklist below).
- Everyone just clones the same read-only repository. Result sharing is done via a
  shared W&B, not git.
- For people who fall behind, the design supports a git tag at the start of each Step
  (`step-01-start` … `step-08-start`). On top of that, each Step's scripts are
  **self-contained** and do not depend on the previous Step's outputs (a second safety
  net so people can catch up even without the tags).

---

## ⚠️ Pre-event checklist (must be verified on the real system)

The author has neither a GPU nor access to Miyabi and has not run anything live. The
items below **must be verified on the real system before the event**. The relevant
scripts carry `# TODO(miyabi)` / `# TODO(lerobot)` comments.

- [ ] **Miyabi queue names** (`QUEUE_NAME`) — exact names for the GH200 compute queue and the interactive/debug queue
- [ ] **Billing group format** (`GROUP`) and the `#PBS` flag to specify it (`-P`, `--group`, etc.)
- [ ] **`module` names** — the exact `module load` name for Apptainer/Singularity
- [ ] **PBS directive syntax** — node/GPU/walltime specification (`-l select=...:ngpus=...`, etc.) is correct for Miyabi
- [ ] **Apptainer build runs** — `env/apptainer.def` actually builds from the NGC aarch64 base, and `lerobot[extras]` resolves on aarch64
- [ ] **HF pre-download size** — actual size of the dataset/checkpoint vs. free space in the `HF_HOME` shared area
- [ ] **`lerobot-train` / `lerobot-eval` arguments** — confirm arg names (`--batch_size` / `--steps` / `--policy.device` / `--wandb.enable`, etc.) via the v0.5.1 `--help`
- [ ] **LIBERO simulation dependencies** — the extras / env vars needed for evaluation are available on the compute node
- [ ] **Shared W&B offline behavior** — if the compute node is offline, whether `WANDB_MODE=offline` + a later sync is required
