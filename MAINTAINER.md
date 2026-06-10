# MAINTAINER — operational guide for organizers

This repository is the "single source of truth for the runnable code". It is designed
so that **each time you run the event, in principle you only update `config.env`
(each participant edits it) and the Notion side**. The code itself stays stable.

## 1. LeRobot pin-tag policy

- `README.md` records the **tag name (`v0.5.1`) and commit hash**.
- Keep `env/apptainer.def`'s `From:` / install lines / `%labels` in sync with the same version.
- Steps to bump the version:
  1. Check argument changes via `lerobot-train` / `lerobot-eval` `--help` for the new tag.
  2. Re-check arg drift in the `*.sh` files against the new version (rehearse
     `train.sh` / `eval.sh` on CPU with the synthetic dataset).
  3. Resolve the `# TODO(lerobot)` spots in `env/apptainer.def` (NGC tag / extras / LIBERO).
  4. Update the tag/hash table in `README.md` and the `%labels` in `apptainer.def`.
  5. Rebuild the image and confirm at minimum that import and `--help` work.

## 2. `step-XX-start` tags (rescue for people who fall behind)

The design supports cutting a git tag at the start of each Step (safety net #1).

```bash
# Example: tag the "start" of each Step on main
git tag step-01-start <commit>
git tag step-02-start <commit>
# ... up to step-08-start
git push origin --tags
```

- A participant who falls behind can catch up with `git checkout step-05-start`.
- **Safety net #2**: each Step's scripts are **self-contained** and do not depend on
  the previous Step's outputs (data is fetched from HF, output dirs are independent).
  A two-layer design so people can catch up even without the tags.
- For the Step ↔ directory mapping, see the table in `README.md`.

## 3. Shared storage layout (keep heavy data off the ~24 GB personal quota)

Each participant has only ~24 GB of personal space — far too small for the 15 GB image
or the ~33 GB LIBERO dataset. So everything heavy lives under the group share
`/work/gw13/share/handson` (`SHARED_DIR`), in two tiers:

```
/work/gw13/share/handson/        SHARED_DIR — organizer stages ONCE, read by all
├── images/lerobot-v0.5.1.sif    APPTAINER_IMAGE  (built once; nobody rebuilds)
├── libero/                      LIBERO_ROOT      (~33 GB dataset, read-only)
├── torch/                       TORCH_HOME       (ResNet18 backbone cache)
├── hf_home/                     HF_HOME          (pre-downloaded repos, read)
└── <username>/                  USER_DIR — each participant's WRITABLE area
    └── outputs/                 OUTPUT_DIR       (checkpoints, eval, W&B)
```

`config.env` derives `USER_DIR=${SHARED_DIR}/${USER}` and `mkdir -p`s it, so each
participant gets their own writable subdir just by `source config.env`. Compute jobs
read the shared tier and write only under `USER_DIR` — their container `--home` is set
to `OUTPUT_DIR`, because the shared `HF_HOME` is read-only for non-owners.

### One-time setup (organizer, on the login node)

```bash
source config.env

# 1. Make the share group-writable + sticky so participants can self-create USER_DIR
#    (sticky bit: nobody can delete another person's subdir). Run as the share owner.
chmod 3770 "${SHARED_DIR}"

# 2. Build the image ONCE into the shared area (APPTAINER_IMAGE points there).
bash env/build_image.sh

# 3. Stage the shared LIBERO dataset + ResNet18 backbone (login node has internet).
bash 04_policy_training/download_libero.sh

# 4. Pre-download the eval checkpoint / generic dataset into the shared HF cache.
bash env/predownload_hf.sh
```

Participants then only: clone the repo, `cp config.env.example config.env`, edit the
day-of values, `source config.env` (which creates their `USER_DIR`), and submit jobs.
They build **no** image and download **no** dataset.

## 4. What to do each time you run the event (minimum)

- [ ] Update the day-of info on Notion (queue names, billing number, W&B project/entity, data repo).
- [ ] Tell participants to copy `config.env.example` → `config.env` and edit it (the repo is read-only).
- [ ] Do the one-time **shared setup** (section 3): `chmod` the share, build the image
      into it, and stage the dataset / HF cache. Participants build/download nothing.
- [ ] If needed, re-cut the `step-XX-start` tags at this round's HEAD.

## 5. Unverified points (handover)

See the "Pre-event checklist" at the end of `README.md` and the `# TODO(miyabi)` /
`# TODO(lerobot)` comments in each script. **Do not erase a TODO with a fabricated
value.** As soon as you can confirm it on the real Miyabi, replace the TODO with the
real value and shrink the checklist.
