# 01_hpc — HPC Overview & Warm-Up

## What is HPC / Miyabi overview

You can't train these models on a laptop. High Performance Computing (HPC) = a shared cluster of powerful nodes you reach over the network and use through a job scheduler.

**Miyabi** — operated by JCAHPC (Univ. of Tokyo + Univ. of Tsukuba):

- Miyabi-G: 1,120 nodes, each one **NVIDIA GH200 Grace-Hopper** (CPU+GPU fused).
- CPU is aarch64 (ARM) — not x86. This matters for software.
- Scheduler: PBS Pro. You submit jobs; you don't run on the login node.

```
        your laptop
            │  ssh
            ▼
   ┌───────────────────┐  qsub  ┌──────────────────────────────┐
   │   login node      │───────▶│      PBS Pro scheduler       │
   │  edit / build /   │        │   (queues / resource groups) │
   │  pre-download     │        └───────────────┬──────────────┘
   └───────────────────┘                        │ dispatches
            ▲                                    ▼
            │ read/write          ┌──────────────────────────────┐
   ┌────────┴──────────┐  mount   │  compute nodes (Miyabi-G)    │
   │  shared storage   │◀────────▶│  1,120 × NVIDIA GH200        │
   │  /work/gw13/$USER │          │  Grace-Hopper · aarch64      │
   └───────────────────┘          │  (offline: no internet)      │
                                  └──────────────────────────────┘
```

## Developing on HPC

The login node is shared by everyone — never run heavy compute there. Real work happens on a compute node, which you get through the scheduler.

```
 LOGIN NODE (shared, has internet)        COMPUTE NODE (yours, offline)
 ─────────────────────────────────        ─────────────────────────────
 - edit code, git clone                    • run training / heavy compute
 - build the Apptainer image      ───────▶ • use the GPU (nvidia-smi)
 - pre-download HF datasets                • read them from shared storage
 - submit jobs (qsub)                      • NO internet here
 ✗ never run heavy compute here

 Two ways to get a compute node:
   interactive   qsub -I ...    → a shell on a node now   (dev / warm-up)
   batch         qsub job.pbs   → queued, runs later      (long training)
```

Two consequences you will feel today:

- **Containers.** We run inside a prebuilt Apptainer image (Docker isn't available on HPC). The image is built once on the login node from an aarch64 base. *(What a container is, and why Apptainer — see the next section.)*
- **Compute nodes are offline.** Datasets and checkpoints are pre-downloaded to shared storage on the login node; jobs read them with `HF_HUB_OFFLINE=1`.
- A GH200 can be split into **MIG** slices, so many of us share GPUs during the class.

## Containers on HPC — what they are, and why Apptainer (not Docker)

A **container** packages an application *together with everything it needs to run* — the right Python, the right CUDA libraries, system packages, your dependencies — into one isolated, reproducible unit. Unlike a virtual machine it shares the host's kernel, so it starts in milliseconds and runs at native speed. The key idea: **the environment lives inside the container, not on the node**, so it behaves the same on your laptop, the login node, and any of the 1,120 compute nodes.

Why containers aren't optional on HPC:

- **No root.** You can't `apt-get install` system libraries on a shared cluster — the container brings them with it.
- **Pinned & reproducible.** The same image gives the same environment on every node, every run. No "works on my node" surprises.
- **Offline-friendly.** Compute nodes have no internet, so you can't `pip install` at run time. Everything is baked into the image *once*, on the login node (which does have internet).

### Docker vs Apptainer

You've probably met **Docker**. Docker relies on a **root-privileged background daemon** (`dockerd`) — on a shared, multi-user supercomputer that is a security non-starter, so Docker isn't available here. HPC uses **Apptainer** (formerly *Singularity*) instead:

```
            DOCKER  (laptops / CI)            APPTAINER  (HPC)
       ──────────────────────────       ──────────────────────────
 runs   a root daemon (dockerd)          as *you*, no daemon, no root
 image  layers in a hidden store         one visible file:  image.sif
 mounts explicit  -v  flags              your $HOME & cwd auto-mounted
 GPU    --gpus                           --nv   (borrows the host driver)
```

- An Apptainer image is a **single file**, `image.sif`, sitting on shared storage. One file = your whole environment: copy it, version it, share it with the class.
- You can still **reuse the Docker ecosystem**: `apptainer pull docker://nvcr.io/...` converts a Docker / NGC image into a `.sif`. That's essentially what `env/build_image.sh` does for us.

### Two rules that bite people

1. **The image must match the CPU architecture.** Miyabi-G is **aarch64 (ARM)**, so the base image and every wheel inside it must be aarch64. An x86 Docker image will *not* run here.
2. **`--nv` means host driver + container CUDA.** The NVIDIA *driver* comes from the host node; the CUDA *toolkit* and PyTorch come from *inside* the image. That split is why a brand-new host driver happily runs a container built against an older CUDA — and why forgetting `--nv` makes the GPU vanish.

```
   apptainer exec --nv  $APPTAINER_IMAGE  python train.py
                   │           │                 └─ your code + PyTorch + CUDA toolkit   ← from the .sif
                   │           └─ the environment, one file on /work                     ← you built this
                   └─ "lend me the host node's GPU + driver"                             ← from the node
```

## Submitting jobs

Three commands are 90% of daily life:

- **qsub** — submit a job
- **qstat** — check status
- **qdel** — cancel a job

A PBS script = scheduler directives (`#PBS ...`) + the commands to run.

```
 #!/bin/bash
 #PBS -q <resource group>    <- which queue (a MIG group for the class)
 #PBS -W group_list=gw13     <- billing group   (NOTE: -W group_list=, NOT -P)
 #PBS -l select=1            <- resources (1 node / 1 MIG slice; no "ngpus=" on Miyabi)
 #PBS -l walltime=00:15:00   <- max run time (job is killed if exceeded)
 #PBS -N warmup              <- job name
 ==========================   ^ above: scheduler directives
 cd $PBS_O_WORKDIR           <- start where you ran qsub
 module load <apptainer>     v below: the commands that actually run
 apptainer exec --nv image.sif nvidia-smi
```

Lifecycle and where output goes:

```
 qsub job.pbs --> [Q] queued --> [R] running --> [done]
      |  returns a job id (e.g. 12345.opbs)         |
      v                                             v
 qstat -u $USER   watch status               warmup.o12345  (stdout)
 qdel 12345       cancel                      warmup.e12345  (stderr)
```

Billing & queues — your group is gw13. The exact queue (resource group) names depend on your allocation; you list them with `qstat --rsc`. For the class we use a MIG resource group.

## Job submission practice

Goal: get onto a real GH200 node and confirm your environment works — before anything depends on it. We use an interactive job so feedback is immediate.

Step 1 — log in and get the code (login node):

```bash
ssh -l <your-account> miyabi-g.jcahpc.jp     # exact host from the day-of handout
cd /work/gw13/$USER                          # your work directory (NOT $HOME)
git clone https://github.com/makolon/lerobot-handson.git
cd lerobot-handson
```

Step 2 — see which queues you can use:

```bash
qstat --rsc        # find your compute (MIG) and interactive resource groups
```

```
[b20066@miyabi-g3 b20066]$ qstat --rsc
SYSTEM: Miyabi-G
QUEUE                     STATUS                 NODE
debug-g                   [ENABLE, START]          48
short-g                   [ENABLE, START]          24
regular-g
  |-- small-g             [ENABLE, START]        1024
  |-- medium-g            [ENABLE, START]        1024
  |-- large-g             [ENABLE, START]        1024
  `-- x-large-g           [ENABLE, START]        1024
interact-g
  |-- interact-g_n1       [ENABLE, START]          48
  `-- interact-g_n8       [ENABLE, START]          48
coupler-g                 [ENABLE, START]        1024

QUEUE                     STATUS                 NODE  MIG
debug-mig                 [ENABLE, START]          12    4
short-mig                 [ENABLE, START]           4    4
regular-mig               [ENABLE, START]           8    4
interact-mig
  `-- interact-mig_g1     [ENABLE, START]          12    4

SYSTEM: Miyabi-C
QUEUE                     STATUS                 NODE
debug-c                   [ENABLE, START]          16
short-c                   [ENABLE, START]           6
regular-c
  |-- small-c             [ENABLE, START]         168
  |-- medium-c            [ENABLE, START]         168
  `-- large-c             [ENABLE, START]         168
interact-c
  |-- interact-c_n1       [ENABLE, START]          16
  `-- interact-c_n2       [ENABLE, START]          16
coupler-c                 [ENABLE, START]         168

SYSTEM: Prepost
QUEUE                     STATUS                 NODE
prepost                   [ENABLE, START]           3
```

Step 3 — grab an interactive shell on a compute node (instant, no batch wait):

```bash
qsub -I -q $QUEUE_NAME_INTERACTIVE -W group_list=gw13 -l select=1 -l walltime=00:15:00
# $QUEUE_NAME_INTERACTIVE: queue name to pass to qsub on Miyabi-G
#   Interactive (full GPU node) : interact-g    <- a whole GH200, MIG disabled (→ interact-g_n1)
#   Interactive (1/4 MIG slice) : interact-mig  <- one MIG instance           (→ interact-mig_g1)
#   Full GPU node (batch)       : debug-g / short-g / regular-g
#   MIG (batch)                 : debug-mig / short-mig / regular-mig
#   Check the queues your group can actually submit to with `qstat --rsc`.
```

Step 4 — you're now ON a compute node. Confirm it:

```bash
hostname      # a compute node (e.g. mg0001), not the login node
uname -m      # aarch64  <- this is an ARM machine
nvidia-smi    # your GH200 / MIG slice should appear
exit          # leave the compute node when done looking around
```

Step 5 — set up config and build the container image (back on the login node):

```bash
cd /work/gw13/$USER/lerobot-handson
cp config.env.example config.env
# Fill in config.env (**W&B + Hugging Face tokens**) before sourcing it.
source /work/gw13/$USER/lerobot-handson/config.env
module load apptainer/1.3.5
bash env/build_image.sh       # build the aarch64 Apptainer image
echo "$APPTAINER_IMAGE"       # sanity check: a real .sif path, not empty
```

Step 6 — now do it the batch way (the real workflow). Submit the tiny warm-up job, watch it, read its log:

```bash
# warmup.pbs hard-codes its #PBS directives (queue debug-g, group_list=gw13,
# select=1, walltime=00:10:00) so this first submit is a single command. The
# parametrized "pass -q/-W/-l at submit time" pattern is taught later in train.pbs.
qsub 01_hpc/warmup.pbs          # prints a job id, e.g. 12345.opbs
qstat -u $USER                  # watch Q -> R -> (disappears when done)
cat warmup.o*                   # read the output log (nvidia-smi, torch check)
```

If `cuda.is_available()` is `True` and the batch log shows the GPU, your environment is ready. If anything failed, paste the error into the shared channel now — this is the moment to catch it.

## Exercise checklist (演習)

Work through "Job submission practice" above. You are done with module 01 when you can tick every box:

- [ ] **Interactive node** — `qsub -I ...` (Step 3) dropped you onto a compute node, and `hostname` shows a compute node (e.g. `mg0001`), not the login node `miyabi-g3`.
- [ ] **Architecture** — `uname -m` prints `aarch64` (this is why the image must be ARM).
- [ ] **GPU visible** — `nvidia-smi` shows your GH200 / MIG slice.
- [ ] **Image built** — `bash env/build_image.sh` finished and `echo "$APPTAINER_IMAGE"` points at a real `.sif` file.
- [ ] **Batch job ran** — `qsub ... 01_hpc/warmup.pbs` returned a job id, you watched it go `Q → R → done` with `qstat -u $USER`, and `warmup.o*` exists.
- [ ] **CUDA from inside the container** — the warm-up log shows `cuda available: True` and a device name. (If it says `CPU only`, you forgot `--nv` or the image is CPU-only — see `env/apptainer.def`.)

The warm-up job itself is [`warmup.pbs`](./warmup.pbs); read it — it is the smallest possible example of the `apptainer exec --nv $APPTAINER_IMAGE ...` pattern every later module reuses.
