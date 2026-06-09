# Exercise — Train SmolVLA on an ALOHA Sim dataset (~20 min)

Work through these in order. Try each one yourself first, then open the Answers to
check. Everything uses only the commands from the `04_policy_training` block.

**Setup:** you have the repo cloned and `config.env` sourced. Your group is `gw13`.
The login node has internet; the compute node does not.

> The answer commands are also captured as runnable scripts in this directory, so you
> can either type the commands yourself or run the scripts:
>
> | File | Role |
> |------|------|
> | [`download_aloha_sim.sh`](./download_aloha_sim.sh) | **login node**: download the ALOHA Sim dataset + `lerobot/smolvla_base` into `$HF_HOME`, warm-load the dataset (Q1) |
> | [`train_smolvla.sh`](./train_smolvla.sh) | SmolVLA config layer over [`../train.sh`](../train.sh) — sets `--policy.path=lerobot/smolvla_base`, `--policy.dtype=bfloat16`, small batch (Q2) |
> | [`train_smolvla.pbs`](./train_smolvla.pbs) | batch PBS wrapper (the interactive answer's submit-and-walk-away equivalent) |
>
> `train_smolvla.sh` does **not** re-implement the training command — it forces
> SmolVLA-appropriate values and `exec`s the shared [`../train.sh`](../train.sh) body,
> exactly like [`../train_libero.sh`](../train_libero.sh). One source of truth for the
> `lerobot-train` call.

## Questions

### Q1. Pick and download a dataset

```
Browse the LeRobot datasets on the Hub and download an ALOHA Sim dataset yourself.

1. Open https://huggingface.co/lerobot and find a dataset whose name starts with
   aloha_sim_  (e.g. insertion or transfer_cube, human or scripted).
2. The compute node has no internet, so download it on the LOGIN node, into $HF_HOME.
3. Also download the SmolVLA base model lerobot/smolvla_base — you fine-tune from it.
```

### Q2. Train SmolVLA

```
Fine-tune SmolVLA on the dataset you just downloaded, and confirm it trains.

1. Get an interactive GPU and enter the container.
2. Run a short lerobot-train job (a few hundred steps) that fine-tunes
   lerobot/smolvla_base on your dataset.
3. Confirm that train/loss drops and a checkpoint is written.

You are checking that SmolVLA trains end-to-end, not training a strong policy.

†If the interactive queue is busy you may wait for the GPU. A short run is
enough — you do not need the loss to converge.
```

## Answers

### A1. Download (login node, has internet)

Pick any `aloha_sim_*` dataset — here `lerobot/aloha_sim_insertion_human` (50 episodes,
bimanual ALOHA, top camera). Download on the login node:

```bash
cd /work/gw13/$USER/lerobot-handson && source config.env
export DATASET=lerobot/aloha_sim_insertion_human     # the one you picked

# dataset + SmolVLA base model -> cached under $HF_HOME
apptainer exec $APPTAINER_IMAGE huggingface-cli download --repo-type dataset $DATASET
apptainer exec $APPTAINER_IMAGE huggingface-cli download lerobot/smolvla_base
```

Or run the script (also warm-loads the dataset, and uses the same `--home`/no-xet
hardening as `download_libero.sh` to avoid filling the small `/home` quota):

```bash
source config.env
SMOLVLA_DATASET=lerobot/aloha_sim_insertion_human \
  bash 04_policy_training/exercise/download_aloha_sim.sh
```

### A2. Fine-tune SmolVLA

**Interactive** — get a GPU, enter the container, and fine-tune:

```bash
qsub -I -q $QUEUE_NAME_INTERACTIVE -W group_list=gw13 -l select=1 -l walltime=01:00:00
module load apptainer/1.3.5
cd /work/gw13/$USER/lerobot-handson && source config.env
apptainer shell --nv $APPTAINER_IMAGE  # now inside the container, on the GPU
```

```bash
# inside the container, on the GPU
export HF_HUB_OFFLINE=1        # use the files you downloaded on the login node
export WANDB_MODE=offline
export DATASET=lerobot/aloha_sim_insertion_human

lerobot-train \
  --dataset.repo_id=$DATASET \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=4 \
  --steps=500 \
  --output_dir=$OUTPUT_DIR/smolvla_aloha --job_name=smolvla_aloha \
  --wandb.enable=true --wandb.project=$WANDB_PROJECT \
  --policy.push_to_hub=false
```

Equivalently, from inside the interactive container session, run the script (it forces
the same arguments via `../train.sh`):

```bash
SMOLVLA_DATASET=$DATASET bash 04_policy_training/exercise/train_smolvla.sh
```

**Batch** — submit-and-walk-away (same body, offline binds handled for you):

```bash
source config.env
qsub -q "$QUEUE_NAME" -W group_list="$GROUP" \
     -l select=1 -l walltime="$WALLTIME" \
     04_policy_training/exercise/train_smolvla.pbs
qstat                                  # Q -> R -> done
cat smolvla_aloha_train.o<jobid>       # read THIS job by id
```

You are done when `train/loss` drops and a checkpoint appears under:

```
$OUTPUT_DIR/smolvla_aloha/checkpoints/last/pretrained_model/
```

### Notes

- `--policy.path=lerobot/smolvla_base` fine-tunes the pretrained SmolVLA. (To train
  from scratch instead: `--policy.type=smolvla --policy.load_vlm_weights=true`.)
- Unlike ACT, SmolVLA reads the `task` instruction — it is a VLA, so the dataset's
  language field is used.
- SmolVLA is larger than ACT; keep `--batch_size` small (4) to stay within memory.
- If `lerobot/smolvla_base` fails to load, the container may be missing SmolVLA's extra
  dependencies (`lerobot[smolvla]`).
