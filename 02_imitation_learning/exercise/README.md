# Exercise — Imitation Learning + LeRobot (~10 min)

Work through these in order. Try each one yourself first, then open the Answers to check.
The concept questions come straight from the Section 2 chapter; the hands-on uses only the
`Try it` commands you already saw.

**Setup:** the repo is cloned, `config.env` is sourced, and the shared image is at
`$APPTAINER_IMAGE`. Everything here runs on the login node, inside the container — no GPU,
no job needed.

> The hands-on (Q5) is also captured as a runnable script, and the interactive,
> plot-everything version lives in the notebook:
>
> | File | Role |
> |------|------|
> | [`read_dataset.py`](./read_dataset.py) | **answer to Q5** — prints fps / episodes / every feature's shape, flags the camera keys and the action dimension |
> | [`../explore.ipynb`](../explore.ipynb) | the fuller walkthrough — `delta_timestamps` window, one DataLoader batch, a plotted frame, and three `🔧 Try it` extensions |

## Questions

### Q1. Behavior cloning

```
Which statement about behavior cloning (BC) is correct?

(a) BC needs a reward function and a lot of trial and error.
(b) BC treats each (observation, action) pair from the demos as a
    supervised example, and trains the policy to output the expert's action.
(c) BC runs the policy on the real robot during training to collect data.
(d) BC only works if there is exactly one correct action per observation.
```

### Q2. Compounding error

```
At test time a BC policy makes a small mistake, lands in a state the demos
never showed, and gets worse from there. What is this, and what helps?

(a) Multimodality — fix it with a bigger learning rate.
(b) Compounding error — predict a chunk of actions, and use more/broader data.
(c) Overfitting — fix it by training for many more steps.
(d) Compounding error — fix it by lowering the success-rate threshold.
```

### Q3. Multimodality

```
One observation, two equally good actions: go left OR right around the cup.
You train with a plain MSE regression to a single action. What happens,
and which policies avoid it?

(a) It picks left or right at random — fine either way.
(b) It averages left and right into a middle action that hits the cup;
    ACT (latent z) and Diffusion Policy (sampling) are built to avoid this.
(c) It learns both and always does the safer one; no special policy needed.
(d) Nothing — MSE represents multiple modes by default.
```

### Q4. Which policy

```
You have ONE task, clean demos, and you want a finished, working policy as
fast as possible today. Which one, and why?

(a) π0 — a pretrained generalist always wins.
(b) GR00T N1.5 — we need cross-embodiment.
(c) ACT — trained from scratch for a single task, fast and small, no language needed.
(d) Diffusion Policy — because you must have language conditioning.
```

### Q5. Hands-on — read a real LeRobotDataset

```
Open the dataset we will train on and read its structure off the real object.
On the login node, inside the container (no GPU):

1. Print the dataset's fps, its number of episodes, and the shape of every feature.
2. From that output, answer:
     - what is the dimension of the `action` vector?
     - which keys are camera images?
     - which two of the five parts (DATA / ENVIRONMENT) do these keys belong to?
3. List the pre-downloaded repos in $HF_HOME/hub. Which look like DATASET repos,
   and which look like MODEL (checkpoint) repos?
```

## Answers

### A1. `(b)`

BC = supervised learning on (observation, action) pairs from the demos. (a) describes RL
(reward + trial and error). (c) BC is offline — no robot or simulator while training. (d)
is false — handling several correct actions (multimodality) is exactly one of BC's hard
problems.

### A2. `(b)`

This is **compounding error** — the policy only saw expert states, so one small drift
lands it off-distribution and the error feeds on itself. What helps: more and broader
data, and predicting a chunk of actions (commit to a short plan instead of reacting
blindly every step). Not a learning-rate or step-count knob; (d) just hides the failure.

### A3. `(b)`

A single MSE target averages the modes — left and right average into a straight-into-the-
cup move. ACT represents the variation with its CVAE latent `z`; Diffusion Policy samples
from noise, so both keep the modes apart instead of averaging them.

### A4. `(c)` ACT

One task + clean demos + fastest finished job = ACT: small, one forward pass per chunk, no
language needed, trained from scratch for this one task. The VLAs (a, b) are heavy
generalists you fine-tune. (d) is wrong reasoning — Diffusion Policy has no language in its
base form, and this task needs none.

### A5. Read it off the object

Run the answer script (reads `$DATA_REPO` from the pre-downloaded cache):

```bash
source config.env
apptainer exec $APPTAINER_IMAGE python 02_imitation_learning/exercise/read_dataset.py

# list the pre-downloaded repos
ls $HF_HOME/hub
```

How to read it:

- **action dimension** = the shape printed on the `action` row (e.g. `(7,)` for LIBERO =
  6-DoF delta pose + gripper; an ALOHA bimanual dataset is `(14,)` = 2 arms × 7).
- **camera images** = the keys under `observation.images.*` (e.g.
  `observation.images.image`). Stored as `(H, W, 3)`, but a decoded frame comes back
  channels-first `(3, H, W)` — what PyTorch policies expect.
- **the five parts**: `observation.*` and `action` are the **DATA** (the recorded demos);
  the same `observation` / `action` format is the contract the **ENVIRONMENT** (simulator
  or robot) must match at eval time.
- **Hub two faces** (`ls $HF_HOME/hub`): a `datasets--…` entry is a `LeRobotDataset` used
  to **TRAIN** (e.g. `datasets--HuggingFaceVLA--libero`); a `models--…` entry holds trained
  policy weights used to **EVALUATE** / fine-tune (the policy name in the repo id is the
  giveaway).

Done early? Open [`../explore.ipynb`](../explore.ipynb) and do its three `🔧 Try it`
extensions (change the `delta_timestamps` window, compute the mean action of episode 0, and
add an image key to the window so the tensor gains a leading `T`).
