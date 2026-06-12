# Exercise — Dataset Conversion (LIBERO → LeRobot) (~15 min)

Work through these in order. Try each one yourself first, then open the Answers to
check. Everything uses only the commands from the `03_dataset_conversion` block.

**Setup:** the repo is cloned and `config.env` is sourced. One LIBERO task file is
staged for you in the shared area — its path is `$LIBERO_TASK_HDF5` (e.g. one task from
LIBERO-Object). It was staged once on the login node with
[`../download_libero_hdf5.py`](../download_libero_hdf5.py); you only read it here.

> The convert step makes `datasets` write a cache under `HF_HOME`, and the **shared**
> `HF_HOME` is read-only for you. Point it at your personal area first (apptainer's
> `--bind` also needs the path to exist):
> ```bash
> export HF_HOME="${USER_DIR}/hf_home" && mkdir -p "$HF_HOME"
> ```

We convert a real LIBERO task file — the same simulator you evaluate on in Section 5.
LIBERO's raw `.hdf5` is in the robomimic/robosuite format: one group per demo, camera
streams + proprioception under `obs/`, and the actions next to it.

**Target LeRobot schema** (what the LIBERO policy expects):

```
 observation.images.image    agentview (third-person) camera          video -> mp4
 observation.images.image2   eye_in_hand (wrist) camera               video -> mp4
 observation.state           8-D = ee_pos(3) + ee_ori(3) + gripper(2)  parquet
 action                      7-D (6-DoF delta pose + gripper)          parquet
 task                        the language instruction (every frame)
```

> Files in this directory:
>
> | File | Role |
> |------|------|
> | [`convert_libero_hdf5.py`](./convert_libero_hdf5.py) | **the answer** to Q3 — converts the LIBERO `.hdf5` into a `LeRobotDataset`. Try to write your own first; this is the reference. |
> | [`test_conversion.py`](./test_conversion.py) | reloads your dataset and asserts it is correct (prints `OK`). Use it to check your own attempt. |
> | [`../convert_sample.py`](../convert_sample.py) | the simpler `.npz` template the answer is adapted from. |

## Questions

### Q1. Raw → LeRobot

```
In the LIBERO hdf5 you find:
  data/demo_0/obs/eye_in_hand_rgb   (T, 128, 128, 3)  uint8

In the LeRobotDataset, which key should this become — and how is it stored?

(a) observation.state         -> parquet
(b) observation.images.image  -> mp4   (this is the agentview camera)
(c) observation.images.image2 -> mp4   (this is the wrist camera)
(d) action                    -> parquet
```

### Q2. Building observation.state

```
LIBERO has no single "state" array. Under obs/ you find ee_pos (T,3),
ee_ori (T,3), gripper_states (T,2), joint_states (T,7). The LIBERO policy
wants observation.state = 8-D. How do you build it?

(a) use joint_states (T,7) and pad one zero to reach 8
(b) concatenate ee_pos(3) + ee_ori(3) + gripper_states(2) -> (8,)
(c) use ee_pos(3) + ee_ori(3) only -> (6,), the policy will pad it
(d) any 8 numbers work — the policy normalizes them anyway
```

### Q3. Convert it yourself (the main task)

```
Write your own convert_libero_hdf5.py to convert the LIBERO task file into a
LeRobotDataset, then run it and make the test print OK.

Raw layout (robomimic/robosuite format):
  data/demo_N/obs/agentview_rgb    (T,H,W,3) uint8  -> observation.images.image
  data/demo_N/obs/eye_in_hand_rgb  (T,H,W,3) uint8  -> observation.images.image2
  data/demo_N/obs/ee_pos           (T,3) float            ┐
  data/demo_N/obs/ee_ori           (T,3) float  axis-angle ├─ concat -> observation.state (8,)
  data/demo_N/obs/gripper_states   (T,2) float            ┘
  data/demo_N/actions              (T,7) float      -> action
  task (language) -> from the file name, or data.attrs["problem_info"]

Two quirks to handle:
  - robosuite/LIBERO renders images upside-down -> flip vertically (img[::-1]).
  - there is no clean fps attribute; LIBERO control runs at 20 Hz -> fps = 20.
```

## Answers

### A1. `(c)`

`eye_in_hand_rgb` is the wrist camera, so it becomes `observation.images.image2` and is
video-encoded into an mp4. The third-person `agentview_rgb` becomes
`observation.images.image`. Both go to `videos/.../*.mp4`; state and action go to
`data/*.parquet`; the task string goes to `meta/tasks.parquet`.

### A2. `(b)`

Build the 8-D state by concatenating `ee_pos(3) + ee_ori(3) + gripper_states(2)`. That is
the proprioception the LIBERO policy was trained with (end-effector position + axis-angle
orientation + gripper). `joint_states` is a different 7-D space (a), 6-D is the wrong size
(c), and (d) is false — the keys and order are baked into the normalization stats, so the
layout must match.

### A3. Convert, then test

```bash
source config.env
export HF_HOME="${USER_DIR}/hf_home" && mkdir -p "$HF_HOME"   # writable cache for the reload

# 1) inspect first (key names/shapes can vary slightly by LIBERO version)
apptainer exec --bind "$HF_HOME:$HF_HOME" --env "HF_HOME=$HF_HOME" $APPTAINER_IMAGE python - <<'PY'
import os, h5py
f = h5py.File(os.environ["LIBERO_TASK_HDF5"], "r")
data = f["data"]
print("episodes:", len(data.keys()), "| data attrs:", list(data.attrs))
d0 = data[sorted(data.keys())[0]]
d0.visititems(lambda n, o: print(" ", n, getattr(o, "shape", ""), getattr(o, "dtype", "")))
PY

# 2) convert (your own script, or the reference answer in this dir)
apptainer exec --bind "$HF_HOME:$HF_HOME" --env "HF_HOME=$HF_HOME" $APPTAINER_IMAGE \
  python 03_dataset_conversion/exercise/convert_libero_hdf5.py \
    --hdf5 "$LIBERO_TASK_HDF5" --root .out/libero

# 3) test — must print OK
apptainer exec --bind "$HF_HOME:$HF_HOME" --env "HF_HOME=$HF_HOME" $APPTAINER_IMAGE \
  python 03_dataset_conversion/exercise/test_conversion.py \
    --root .out/libero --hdf5 "$LIBERO_TASK_HDF5"
```

The reference [`convert_libero_hdf5.py`](./convert_libero_hdf5.py) is the same template as
[`../convert_sample.py`](../convert_sample.py): define `features`, `create`, then
`add_frame` → `save_episode` per demo, and one `finalize()` at the end. The test reloads
your dataset and checks: it loads, `num_episodes` matches the hdf5, the four features
exist, `fps == 20`, a sample frame has state `(8,)`, action `(7,)`, channels-first images
`(3, H, W)`, and every frame carries a non-empty `task` string. All green → it prints `OK`.

### Notes

- Key names vary a little by LIBERO version — that is why you **inspect first**. If your
  file has `ee_states` (6-D, pos+ori) instead of separate `ee_pos` / `ee_ori`, split it
  (`ee_states[:, :3]`, `ee_states[:, 3:6]`) or use it as the first 6 dims of the state.
- Two things silently break a conversion: forgetting the `"task"` key in every
  `add_frame`, and forgetting `finalize()` (the parquet stays open and the dataset won't
  load). The reference answer does both correctly.
