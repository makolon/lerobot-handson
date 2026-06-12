# Exercise — Warm up on HPC + Containers (~5 min)

Work through these in order. Try each one yourself first, then open the Answers to check.
Everything uses only the commands from the `01_hpc` block.

**Setup:** you have the repo cloned and `config.env` sourced. Your group is `gw13`.

> The Q4 answer is the runnable job in this directory:
>
> | File | Role |
> |------|------|
> | [`warmup.pbs`](./warmup.pbs) | **answer to Q4** (symlinked to [`../warmup.pbs`](../warmup.pbs)) — a minimal batch job that prints the host and runs `nvidia-smi` inside the container |

## Questions

### Q1. HPC & Miyabi

```
You have just SSH'd into the login node. Which one is OK to do right here?

(a) run a 2-hour python train.py
(b) clone the repo, edit your script, and submit it with qsub
(c) start a process that uses all CPU cores
(d) run a long GPU benchmark
```

### Q2. Docker & Apptainer

```
Why does Miyabi use Apptainer instead of Docker?

(a) Docker cannot use GPUs
(b) Docker needs a root-privileged daemon, which is unsafe on a shared, multi-user system
(c) Docker images are too large
(d) Docker does not support Python
```

### Q3. Jobs & setup

```
Which command correctly starts an interactive session for the class
(group gw13, 15 minutes)?

(a) qsub -I -q <queue> -P gw13 -l select=1 -l walltime=00:15:00
(b) qsub -I -q <queue> -W group_list=gw13 -l select=1 -l walltime=00:15:00
(c) qsub    -q <queue> -W group_list=gw13 -l select=1 -l walltime=00:15:00
```

### Q4. Submit a batch job

```
Write a short batch script that runs on a compute node and checks that the
container can see the GPU.

1. Create a file warmup.pbs with the five directives you need
   (queue, group, resources, walltime, name).
2. Then have it print the host name and run nvidia-smi inside the container.
3. Submit it with qsub.
4. Check its status with qstat.
5. When it leaves the queue, read the output file.

Please use a pre-built Apptainer image.
```

## Answers

### A1. `(b)`

The login node is shared. It is for light prep — editing, cloning, downloading data — and
for submitting jobs with `qsub`. Anything heavy or GPU-bound (a, c, d) must run on a
compute node through a job.

### A2. `(b)`

Docker's `dockerd` runs as root, so giving every user access to it is close to giving them
root on a shared machine. Apptainer runs as you, with no daemon and no root, and its image
is a single `.sif` file. (You can still convert Docker / NGC images with
`apptainer pull docker://...`.)

### A3. `(b)`

(a) sets the group with `-P`. On Miyabi the group is set with `-W group_list=`, not `-P`,
so the job is rejected or charged to the wrong group. (c) has no `-I`, so PBS treats it as
a batch submission and expects a script — it does not open an interactive session. (b) has
both: `-I` for interactive, and `-W group_list=gw13` for the correct group flag.

### A4. The job is [`warmup.pbs`](./warmup.pbs)

```bash
#!/bin/bash
#PBS -N warmup
#PBS -q debug-g
#PBS -W group_list=gw13
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -j oe

cd "${PBS_O_WORKDIR:-$(pwd)}"
source config.env
module load "${APPTAINER_MODULE}"
echo "Running on host: $(hostname)"
apptainer exec --nv "${APPTAINER_IMAGE}" nvidia-smi
```

Submit it, watch it, read its log:

```bash
source config.env
qsub 01_hpc/exercise/warmup.pbs   # prints a job id, e.g. 12345.opbs
qstat                             # Q = waiting, R = running
qdel <jobid>                      # cancel it if it is stuck in Q
cat warmup.o<jobid>               # after it finishes: the host name + nvidia-smi output
```

If `nvidia-smi` lists the GH200 in the log, the container sees the GPU and your
environment is ready.
