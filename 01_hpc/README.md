# 1. HPC Overview & Warm-Up

## What is HPC / Miyabi overview

You can't train these models on a laptop. High Performance Computing (HPC) = a shared cluster of powerful nodes you reach over the network and use through a job scheduler.

**Miyabi** is operated by **JCAHPC** (the Joint Center for Advanced High Performance Computing), run jointly by the University of Tokyo and the University of Tsukuba. At 80.1 PFLOPS it is the largest cluster supercomputer in Japan, and the country's first large-scale general-purpose **GH200** system.

It has two compute partitions:

|  | Miyabi-G *(we use this today)* | Miyabi-C |
| --- | --- | --- |
| Nodes | 1,120 | 190 |
| Per node | 1× NVIDIA GH200 Grace-Hopper | 2× Intel Xeon Max 9480 (CPU, on-package HBM) |
| CPU | 72-core Grace, **aarch64 (ARM)** | x86-64 |
| GPU | Hopper (H100-class) HBM3 — `nvidia-smi` reports "GH200 120GB" | none |
| Peak (DP) | 78.8 PFLOPS | 1.3 PFLOPS |
| For | **GPU training / inference** | CPU-only codes |

Everything in this lecture runs on **Miyabi-G** (the GH200 GPU nodes); Miyabi-C is the CPU-only partition for codes that don't use a GPU.

```
        your laptop
            │  ssh
            ▼
   ┌───────────────────┐  qsub   ┌──────────────────────────────┐
   │   login node      │───────▶ │      PBS Pro scheduler       │
   │  edit / build /   │         │   (queues / resource groups) │
   │  pre-download     │         └───────────────┬──────────────┘
   └───────────────────┘                         │ dispatches
            ▲                                     ▼
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

Because the compute nodes are offline, datasets and checkpoints are **pre-downloaded** to shared storage on the login node, and jobs read them with `HF_HUB_OFFLINE=1` (see `env/predownload_hf.sh`).

### Editing your code on Miyabi

Your files live on the cluster (`/work/gw13/$USER/...`), so you edit them **in place** on the login node. Two common setups:

**VS Code (Remote - SSH)** — a full IDE (file tree, editor, integrated terminal) while the files stay on Miyabi.

1. **Add Miyabi to `~/.ssh/config`** (on your laptop). The `Control*` lines make you enter the one-time password **only once** per session:

   ```
   Host miyabi
       HostName miyabi-g.jcahpc.jp
       User <your-account>
       ControlMaster auto
       ControlPath ~/.ssh/control-%r@%h:%p
       ControlPersist 8h
   ```

2. In VS Code, install the **Remote - SSH** extension.
3. **Open a terminal and run `ssh miyabi` once**, entering your **one-time password**. This opens the shared master connection that VS Code will reuse.
4. In VS Code: Command Palette (`F1`) → **Remote-SSH: Connect to Host** → `miyabi`. It reuses the connection above, so it won't ask for the OTP again.
5. **Open Folder** → `/work/gw13/<your-account>/lerobot-handson`.

> ⚠️ The VS Code server runs **on the login node**. Use it to edit and browse only — never run training in its terminal. Launch real work with `qsub`.
> (`ControlMaster` is macOS/Linux; native Windows OpenSSH doesn't support it, so you'll re-enter the OTP per connection — or use WSL.)

**Neovim (terminal, lightweight)** — nothing to sync, works over any SSH connection. Since you have **no root**, install your tools in userspace:

1. SSH in: `ssh miyabi` (enter your one-time password).
2. Install a user-level package manager such as **pixi** (no root required):

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

   then install Neovim with it — or run your own bootstrap (e.g. makolon/server-bootstrap), which sets up Neovim and a ready-to-use config in one go.
3. Edit in place:

   ```bash
   cd /work/gw13/$USER/lerobot-handson && nvim .
   ```

Either way the rule is the same: **edit on the login node, compute through the scheduler.**

## Containers on HPC

A **container** is a box that holds a program and everything it needs to run — the right Python, the right CUDA libraries, system packages, and your own dependencies. Everything is inside one box. A container is not the same as a virtual machine: it shares the host's kernel, so it starts very fast and runs at full speed.

The main idea: your setup lives **inside the container, not on the node**. So it works the same way everywhere — on your laptop, on the login node, and on any of the 1,120 compute nodes.

Why we must use containers on HPC:

- **You are not root.** On a shared cluster, you can't run `apt-get install` to add system libraries. The container already has them inside.
- **Always the same.** The same image gives you the same setup on every node, every time. Nothing changes from one run to the next.
- **No internet on compute nodes.** Compute nodes can't reach the internet, so you can't run `pip install` there. You install everything once, on the login node (which has internet), and bake it into the image.

### Docker vs. Apptainer

You've probably met **Docker**. Docker relies on a **root-privileged background daemon** (`dockerd`) — on a shared, multi-user supercomputer that is a security non-starter, so Docker isn't available here. HPC uses **Apptainer** (formerly *Singularity*) instead:

```
              DOCKER (laptops / CI)             APPTAINER (HPC)
──────────────────────────────────────────────────────────────────────────────
run model     containers via a root daemon      containers as your own process
              (dockerd)                          (no daemon, no root)
image         layers in a system store           one file you own
              (/var/lib/docker)                  (image.sif)
host files    isolated; bind with -v             $HOME / cwd / /tmp auto-bound;
                                                  add more with -B
GPU           --gpus (NVIDIA Container Toolkit)   --nv (binds the host driver)
```

- An Apptainer image is a **single file**, `image.sif`, sitting on shared storage. One file = your whole environment: copy it, version it, share it with the class.
- You can still **reuse the Docker ecosystem**: `apptainer build` (or `apptainer pull docker://nvcr.io/...`) turns a Docker / NGC image into a `.sif`. That's the basis of `env/build_image.sh`, which builds from `env/apptainer.def`: it bootstraps an NGC PyTorch image (aarch64) and installs LeRobot on top.

### Two rules that bite people

1. **The image must match the CPU architecture.** Miyabi-G is **aarch64 (ARM)**, so the base image and every wheel inside it must be aarch64. An x86 Docker image will *not* run here.
2. **`--nv` means host driver + container CUDA.** The NVIDIA *driver* comes from the host node; the CUDA *toolkit* and PyTorch come from *inside* the image. That split is why a new host driver happily runs a container built against an older CUDA — and why forgetting `--nv` makes the GPU vanish (`torch.cuda.is_available()` → `False`).

```
   apptainer exec --nv  $APPTAINER_IMAGE  python train.py
                   │           │                 └─ your code + PyTorch + CUDA toolkit   ← from the .sif
                   │           └─ the environment, one file on /work                     ← you built this
                   └─ "lend me the host node's GPU + driver"                             ← from the node
```

## Submitting jobs

A job is a **request to the scheduler (PBS)**: you describe the resources you need and the commands to run, and PBS runs them for you on a **compute node** — not on the login node where you type your commands. You can run a job in one of two ways.

### Interactive vs. batch

|  | **Interactive**  `qsub -I ...` | **Batch**  `qsub job.pbs` |
| --- | --- | --- |
| What you get | A shell on the node, so you can type commands live | It runs on its own; you can close your laptop |
| Best for | Testing, first setup, "can my container see the GPU?" | Real training, long jobs, running many at once |
| Watch out | If your connection drops, the job stops. The slot is reserved (and charged) even while you think | You can't watch it live; you read the output files after it ends |
| Output goes to | Your terminal | `*.o` / `*.e` files |

A good habit: **test interactively first, then put the commands that worked into a batch script.**

Three commands = 90% of daily life:

- **qsub** — submit a job   (prints a job id, e.g. `12345.opbs`)
- **qstat** — check status   (`Q` = queued, `R` = running)
- **qdel** — cancel a job   (`qdel 12345`)

### Batch script

A PBS script has two parts: **directives** (`#PBS ...`, read by the scheduler) and **commands** (a normal shell script, run after you get a node).

```bash
#!/bin/bash
#PBS -q <queue>             # which queue (this sets your GPU slot)
#PBS -W group_list=gw13     # which project to charge (yours is gw13)
#PBS -l select=1            # reserve 1 node = 1 GH200 (its GPU is auto-allocated)
#PBS -l walltime=00:15:00   # max run time; the job is killed when time is up
#PBS -N warmup              # job name (also used to name the output files)

cd "$PBS_O_WORKDIR"
module load apptainer/1.3.5
apptainer exec --nv "$APPTAINER_IMAGE" python train.py     # the actual work
```

> On Miyabi the project is set with `-W group_list=`, **not** PBS Pro's `-P` (Miyabi rejects `-P`). One GH200 node is requested with `-l select=1`; its GPU is allocated automatically, so there is **no** `:ngpus=`.

A note on `walltime`: be careful in both directions. Too short, and the job is killed before it finishes. Too long, and it waits longer for a free slot and uses more of your allocation. Make a good guess, then add a little extra.

Two values you fill in yourself:

- `group_list=gw13` — your project for this class (`$GROUP` in `config.env`).
- `-q <queue>` — the GPU slot you're allowed to use (`$QUEUE_NAME` / `$QUEUE_NAME_INTERACTIVE` in `config.env`). List the queues your group can submit to with `qstat --rsc`, then paste the name in.

### Common #PBS directives

You write these as `#PBS ...` lines in the script (or as flags after `qsub` on the command line). The first five are the ones you almost always set.

| Directive | What it does |
| --- | --- |
| `-q <queue>` | which queue to run in (sets your GPU slot) |
| `-W group_list=gw13` | which project to charge |
| `-l select=1` | how many nodes to reserve (1 node = 1 GH200) |
| `-l walltime=00:15:00` | maximum run time |
| `-N <name>` | job name |
| `-J 1-10` | run as a job array (here, 10 sub-jobs) |
| `-o <file>` / `-e <file>` | rename the output / error file |
| `-j oe` | merge stdout and stderr into one file |
| `-m abe` + `-M <address>` | email you when the job aborts / begins / ends |

### What happens after qsub

By default PBS writes **two** files when the job ends: `<job name>.o<id>` (stdout) and `<job name>.e<id>` (stderr). The `warmup.pbs` here sets `-j oe`, so both streams are **merged into one** `warmup.o<id>`:

```
qsub 01_hpc/warmup.pbs ─► [Q] queued ─► [R] running ─► [done]
                                                          │
                            warmup.o12345 ← stdout+stderr ┘
                            (warmup.pbs uses -j oe → one merged file)
```

Read it with ordinary shell commands — for the warm-up job with id `12345`:

```bash
ls warmup.*               # warmup.o12345   (just one file: -j oe merged the streams)
cat warmup.o12345         # everything the job printed (nvidia-smi + the torch check)
tail -n 20 warmup.o12345  # just the last 20 lines
```

## Job submission practice

Goal: get onto a real GH200 node and confirm your environment works — before anything later depends on it. We use an interactive job first so feedback is immediate.

Step 1 — log in and get the code (login node):

```bash
ssh -l <your-account> miyabi-g.jcahpc.jp    # exact host from the day-of handout
cd /work/gw13/$USER                         # your work directory (NOT $HOME)
git clone https://github.com/makolon/lerobot-handson.git
cd lerobot-handson
```

Step 2 — see which queues you can use:

```bash
qstat --rsc        # list the GPU (full-node) and interactive resource groups
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

`interact-g` / `regular-g` / `interact-mig` are **parent groups** — you submit to a **leaf** queue (e.g. `interact-g_n1`, `small-g`, `interact-mig_g1`). The `-mig` queues hand out a 1/4 GH200 **MIG** slice; today we use **full GH200 nodes** (`config.env` sets `QUEUE_NAME=short-g`, `QUEUE_NAME_INTERACTIVE=interact-g_n1`).

Step 3 — grab an interactive shell on a compute node (instant, no batch wait):

```bash
# interact-g_n1 = a full GH200 node, interactive, walltime <= 02:00:00
# (this is config.env's $QUEUE_NAME_INTERACTIVE — source config.env first, or type it literally)
qsub -I -q interact-g_n1 -W group_list=gw13 -l select=1 -l walltime=00:15:00
# Other queues you could pass instead:
#   Interactive (full GPU node) : interact-g_n1   <- a whole GH200 (1 node)
#   Interactive (1/4 MIG slice) : interact-mig_g1 <- one MIG instance
#   Full GPU node (batch)       : debug-g / short-g / small-g
#   MIG (batch)                 : debug-mig / short-mig / regular-mig
#   Check the queues your group can actually submit to with `qstat --rsc`.
```

Step 4 — you're now ON a compute node. Confirm it:

```bash
hostname      # a compute node (e.g. mg0010), not the login node miyabi-g3
uname -m      # aarch64  <- this is an ARM machine
nvidia-smi    # your GH200 should appear
exit          # leave the compute node when done looking around
```

Step 5 — set up config and build the container image (back on the login node):

```bash
cd /work/gw13/$USER/lerobot-handson
cp config.env.example config.env
# Fill in config.env's day-of values (HF dataset/checkpoint repos, shared W&B
# project/entity from the handout). Tokens are optional — only needed if you push.
# For this warm-up you don't need any of those; the image is all it requires.
source /work/gw13/$USER/lerobot-handson/config.env
bash env/build_image.sh       # build the aarch64 Apptainer image (on the login node — it needs internet)
echo "$APPTAINER_IMAGE"       # sanity check: a real .sif path, not empty
```

`build_image.sh` loads the Apptainer module itself and pulls the NGC base, so it must run where there is internet — the **login node**. Compute nodes are offline.

Step 6 — now do it the batch way (the real workflow). Submit the tiny warm-up job, watch it, read its log:

```bash
# warmup.pbs hard-codes its #PBS directives (queue debug-g, group_list=gw13,
# select=1, walltime=00:10:00, -j oe) so this first submit is a single command.
# The parametrized "pass -q/-W/-l at submit time" pattern is taught later in train.pbs.
qsub 01_hpc/warmup.pbs          # prints a job id, e.g. 12345.opbs
qstat -u $USER                  # watch Q -> R -> (disappears when done)
cat warmup.o*                   # read the merged output log (nvidia-smi + torch check)
```

If `cuda available` is `True` and the log shows the GPU, your environment is ready. If anything failed, paste the error into the shared channel now — this is the moment to catch it. Don't get stuck; just ask.

## Exercise — Warm up on HPC + Containers (~5 min)

Work through these in order. Try each one yourself first, then open the answers to check. Everything uses only the commands from this module.

**Setup:** you have the repo cloned and `config.env` sourced (Steps 1 & 5). Your group is `gw13`.

### Questions

**Q1 — HPC & Miyabi**

```
You have just SSH'd into the login node. Which one is OK to do right here?

(a) run a 2-hour python train.py
(b) clone the repo, edit your script, build the image, and submit it with qsub
(c) start a process that uses all CPU cores
(d) run a long GPU benchmark
```

**Q2 — Docker & Apptainer**

```
Why does Miyabi use Apptainer instead of Docker?

(a) Docker cannot use GPUs
(b) Docker needs a root-privileged daemon, which is unsafe on a shared, multi-user system
(c) Docker images are too large
(d) Docker does not support Python
```

**Q3 — Jobs & setup**

```
Which command correctly starts an interactive session for the class
(group gw13, 15 minutes)?

(a) qsub -I -q <queue> -P gw13 -l select=1 -l walltime=00:15:00
(b) qsub -I -q <queue> -W group_list=gw13 -l select=1 -l walltime=00:15:00
(c) qsub    -q <queue> -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

**Q4 — Submit a batch job**

```
The repo ships 01_hpc/warmup.pbs. To learn the directives, write a minimal
warmup.pbs of your own that runs on a compute node and checks the container
can see the GPU, then compare it with the shipped one.

1. Write warmup.pbs with the five directives you need (queue, group, resources,
   walltime, name), plus -j oe.
2. Have it print the host name and run nvidia-smi inside the container.
3. Submit it with qsub, check status with qstat, and read the output when it
   leaves the queue.

If your job does not start, that is fine. The class shares a small GPU
allocation, so if everyone submits at once your job may sit in [Q] for a while.
You do not need to wait for it to finish — once you have seen it accepted (a job
id) and listed by qstat, you have done the exercise. Cancel a stuck job with
qdel <id> to free the slot for others.
```

### Answers

**A1: (b).** The login node is shared. It's for light prep — editing, cloning, **building or pulling the image** (only the login node has internet), pre-downloading data — and for submitting jobs with `qsub`. Anything heavy or GPU-bound (a, c, d) must run on a compute node through a job.

**A2: (b).** Docker's `dockerd` runs as root, so giving every user access to it is close to giving them root on a shared machine. Apptainer runs as *you*, with no daemon and no root, and its image is a single `.sif` file. (You can still convert Docker / NGC images with `apptainer build` / `apptainer pull docker://...`.)

**A3: (b).**
- (a) sets the project with `-P`. On Miyabi the project is set with `-W group_list=`, not `-P`, so the job is rejected.
- (c) has no `-I`, so PBS treats it as a batch submission and expects a script — it does not open an interactive session.
- (b) has both: `-I` for interactive, and `-W group_list=gw13` for the correct project flag.

**A4:** a minimal version (the shipped `01_hpc/warmup.pbs` is the same, and also adds a `torch.cuda.is_available()` check inside the container):

```bash
#!/bin/bash
#PBS -N warmup              # job name
#PBS -q debug-g             # quick full-node GPU queue (walltime <= 00:30:00)
#PBS -W group_list=gw13     # your project (NOT -P)
#PBS -l select=1            # one GH200 node (GPU auto-allocated; no :ngpus=)
#PBS -l walltime=00:10:00   # ten minutes is plenty
#PBS -j oe                  # merge stdout+stderr into one warmup.o<id>

cd "$PBS_O_WORKDIR"
source config.env           # for $APPTAINER_IMAGE / $APPTAINER_MODULE
module load apptainer/1.3.5
echo "Running on host: $(hostname)"
apptainer exec --nv "$APPTAINER_IMAGE" nvidia-smi
```

Submit and check:

```bash
qsub 01_hpc/warmup.pbs   # prints a job id, e.g. 12345.opbs
qstat -u $USER           # Q = waiting, R = running
qdel 12345               # cancel it if it is stuck in Q
cat warmup.o12345        # after it finishes: host name + nvidia-smi (one file, -j oe)
```

The warm-up job itself is [`warmup.pbs`](./warmup.pbs); read it — it is the smallest possible example of the `apptainer exec --nv "$APPTAINER_IMAGE" ...` pattern every later module reuses.
