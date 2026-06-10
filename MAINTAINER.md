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

## 3. What to do each time you run the event (minimum)

- [ ] Update the day-of info on Notion (queue names, billing number, W&B project/entity, data repo).
- [ ] Tell participants to copy `config.env.example` → `config.env` and edit it (the repo is read-only).
- [ ] Confirm the image build (`env/build_image.sh`) and HF pre-download
      (`env/predownload_hf.sh`) are done in the shared area on the login node.
- [ ] If needed, re-cut the `step-XX-start` tags at this round's HEAD.

## 4. Unverified points (handover)

See the "Pre-event checklist" at the end of `README.md` and the `# TODO(miyabi)` /
`# TODO(lerobot)` comments in each script. **Do not erase a TODO with a fabricated
value.** As soon as you can confirm it on the real Miyabi, replace the TODO with the
real value and shrink the checklist.
