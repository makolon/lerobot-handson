# challenges/debug — Debug the broken jobs (Bonus 1)

## What this is

Here are 4 PBS jobs, each **intentionally broken in exactly one spot**. Submitting them
fails. Your mission is to "read the log, identify the cause, fix it, and resubmit."
It's practice for stepping on the 4 typical stumbling blocks you'll inevitably hit in
real operation, in a safe place.

| File | Symptom hint (what happens) |
|------|-----------------------------|
| `broken_01_oom.pbs` | execution starts but crashes with something GPU-related |
| `broken_02_offline.pbs` | it starts moving but goes to fetch something externally and hangs/times out |
| `broken_03_bind.pbs` | it dies saying a file "doesn't exist" |
| `broken_04_queue.pbs` | the submission doesn't go through at all (or it's killed partway) |

## How to proceed

1. Confirm you have run `source config.env`.
2. Submit one at a time with `qsub challenges/debug/broken_0X_xxx.pbs`.
3. Check status with `qstat`; when it finishes, **read the output log
   (`*.out` / stderr) to the end**.
4. Infer the cause from the error message, fix the file, and resubmit.
5. To get a sense of the fix, compare against the "correct" version in
   `03_train/train.pbs` / `train.sh`.

## Where are the answers?

**The `main` branch contains no answers.** The solutions (cause and fix for each bug) are in:

- the **Notion solution toggle**, or
- the **`solutions` branch** of this repository (fixed versions of `challenges/debug/`
  plus `challenges/debug/SOLUTIONS.md`).

Try to fix them yourself first, grounded in the logs.
