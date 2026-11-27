# Exercise — Evaluate a pretrained policy in simulation (~20 min)

Work through these in order. Try each one yourself first, then open the Answers to check.
Everything uses only the commands from the `05_policy_evaluation` block.

**Setup:** the repo is cloned and `config.env` is sourced. Your group is `gw13`. Both the
login and compute nodes have internet. The shared `$HF_HOME` is read-only for you, so
download into your own cache: `export HF_HOME=$USER_HF_HOME` (= `$USER_DIR/hf_home`).

> ⚠️ **Not every Hub checkpoint runs in this image.** The shared image installs
> `lerobot[aloha,libero]` only. So:
> - **ACT + aloha** and **ACT + libero** → work (ACT needs no extra policy deps).
> - `diffusion_pusht` (`--env.type=pusht`) → needs `gym-pusht`, **not installed**.
> - `pi0` / `pi0.5` LIBERO VLAs → need `lerobot[pi]`, **not installed**.
>
> The answer below defaults to **`lerobot/act_aloha_sim_transfer_cube_human`** on the
> aloha env, which loads here. You can also evaluate your **own** Section 4 ACT-LIBERO
> checkpoint (no download) — see the last note.

> The answer commands are also captured as runnable scripts in this directory:
>
> | File | Role |
> |------|------|
> | [`download_pretrained.sh`](./download_pretrained.sh) | **login node**: download the checkpoint into `$USER_HF_HOME` (Q1) |
> | [`eval_pretrained.sh`](./eval_pretrained.sh) | run `lerobot-eval` on the checkpoint in its env (Q2); supports `DRY_RUN=1` |
> | [`eval_pretrained.pbs`](./eval_pretrained.pbs) | batch wrapper (binds and `HF_HOME` handled for you) |

## Questions

### Q1. Pick and download a checkpoint

```
Browse the LeRobot models on the Hub and download a pretrained policy checkpoint.
Then work out which simulation environment it supports.

1. Open https://huggingface.co/lerobot and pick a policy checkpoint (a MODEL repo,
   not a dataset).
2. Read its model card to find the environment it was trained for. The name is the
   clue:  *_pusht -> pusht,  aloha_sim_* -> aloha,  *_libero_* -> libero.
3. Pre-download it on the LOGIN node into your own cache, $USER_HF_HOME
   (the shared $HF_HOME is read-only).
```

### Q2. Evaluate it

```
Run the checkpoint in its environment and read the success rate.

1. Get an interactive GPU and enter the container.
2. Run lerobot-eval with --policy.path= and the --env.type / --env.task that match it.
3. Read the average success rate, and watch the rollout videos it saves.

The checkpoint and the environment must match — a LIBERO policy cannot run on the
aloha env, because the observation and action spaces differ.

†A small --eval.n_episodes (e.g. 10) is enough to see a number.
```

## Answers

### A1. Download (login node)

Pick a checkpoint whose policy + env are both supported by the image. A safe default is
`lerobot/act_aloha_sim_transfer_cube_human` (ACT on ALOHA sim):

```bash
cd /work/gw13/$USER/lerobot-handson && source config.env
export HF_HOME=$USER_HF_HOME                            # your own writable cache
export CKPT=lerobot/act_aloha_sim_transfer_cube_human   # the one you picked
module load apptainer/1.3.5

apptainer exec --bind "$SHARED_DIR" $APPTAINER_IMAGE hf download $CKPT
```

Or run the script (same `--home`/no-xet hardening as the other download scripts):

```bash
source config.env
CKPT=lerobot/act_aloha_sim_transfer_cube_human \
  bash 05_policy_evaluation/exercise/download_pretrained.sh
```

How to tell which env: the model card lists the training dataset and env, and the
downloaded `config.json` records the input/output features. The repo name is the quick
clue — `aloha_sim_*` → aloha, `*_libero_*` → libero, `*_pusht` → pusht.

### A2. Evaluate (interactive GPU, in the container)

```bash
qsub -I -q $QUEUE_NAME_INTERACTIVE -W group_list=gw13 -l select=1 -l walltime=01:00:00
module load apptainer/1.3.5
cd /work/gw13/$USER/lerobot-handson && source config.env
apptainer shell --nv --bind "$SHARED_DIR" $APPTAINER_IMAGE   # now inside the container, on the GPU
```

```bash
# inside the container, on the GPU
export HF_HOME=$USER_HF_HOME   # the cache you downloaded into in A1
export HF_HUB_OFFLINE=1        # load the checkpoint from that cache only
export MUJOCO_GL=egl           # headless rendering for the simulator
export MUJOCO_EGL_DEVICE_ID=0  # Miyabi: pin the EGL device (CUDA_VISIBLE_DEVICES is a GPU UUID)
export CKPT=lerobot/act_aloha_sim_transfer_cube_human

lerobot-eval \
  --policy.path=$CKPT \
  --policy.device=cuda \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 --eval.batch_size=10 \
  --output_dir=$OUTPUT_DIR/eval_act_aloha
```

Equivalently, run the script (it forces a matching env/task and the same args):

```bash
CKPT=lerobot/act_aloha_sim_transfer_cube_human \
ENV_TYPE=aloha ENV_TASK=AlohaTransferCube-v0 \
  bash 05_policy_evaluation/exercise/eval_pretrained.sh
```

**Batch** — submit-and-walk-away:

```bash
source config.env
CKPT=lerobot/act_aloha_sim_transfer_cube_human ENV_TYPE=aloha ENV_TASK=AlohaTransferCube-v0 \
qsub -q "$QUEUE_NAME" -W group_list="$GROUP" -l select=1 -l walltime="$WALLTIME" \
     -v CKPT,ENV_TYPE,ENV_TASK 05_policy_evaluation/exercise/eval_pretrained.pbs
qstat                                  # Q -> R -> done
cat eval_pretrained.o<jobid>           # read THIS job by id
```

You are done when `lerobot-eval` prints the overall success rate (`pc_success`, in %). It
also saves rollout videos (up to 10 per task, recorded by default):

```
$OUTPUT_DIR/eval_act_aloha/            # the script writes to $OUTPUT_DIR/eval/<checkpoint name>/
  eval_info.json          # per_task / per_group / overall: successes, rewards, video_paths
  videos/<task_group>_<task_id>/eval_episode_*.mp4   # watch a success AND a failure
```

### Notes

- **Match the policy to the env.** For the default ALOHA ACT checkpoint:
  `--env.type=aloha --env.task=AlohaTransferCube-v0` (or `AlohaInsertion-v0`).
- **Verify the command first** without a GPU: `DRY_RUN=1 OUTPUT_DIR=.smoke/out
  bash 05_policy_evaluation/exercise/eval_pretrained.sh` prints the exact `lerobot-eval`
  line without running it.
- **Evaluate your own Section 4 run** instead of a Hub model — no download needed, and it
  matches the image (ACT + libero):
  ```bash
  POLICY_PATH=$OUTPUT_DIR/act_base/checkpoints/last/pretrained_model \
  ENV_TYPE=libero ENV_TASK=libero_object \
    bash 05_policy_evaluation/exercise/eval_pretrained.sh
  ```
  LIBERO runs every task in the suite (10 for `libero_object`), `EVAL_EPISODES` each.
- `--eval.batch_size` must be ≤ `--eval.n_episodes` (lerobot raises otherwise); the script
  defaults the batch size to the episode count.
- If the env fails to load, the container is missing that env extra — see the ⚠️ box above
  for what this image actually ships.
