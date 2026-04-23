# Changelog - 2026-04-17 - Project Cleanup (Entry Points)

## Scope

- Deleted `scripts/run_eval.py`
- Updated `scripts/README.md`
- Updated `PROJECT_STRUCTURE.md`

## What Was Removed

### `scripts/run_eval.py`

- Removed as a redundant evaluation entry wrapper.
- The project already has a unified evaluation entry point: `evaluation/evaluate.py`.

## Why This Is Safe

- Reference check showed `run_eval.py` was not imported or invoked by core runtime code.
- Remaining mentions were documentation-only.
- No training or evaluation loader path depends on this file.

## Entry Point Status After Cleanup

- Training: unified under `training/` core scripts.
- DPO training: `training/dpo_train.py` (single active DPO entry).
- Evaluation: `evaluation/evaluate.py` (single active eval entry).

## Impact

- Fewer duplicate scripts and lower maintenance burden.
- Project behavior unchanged for normal workflows.
- Documentation now matches actual executable entry points.
