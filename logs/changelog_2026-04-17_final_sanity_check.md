# Changelog - 2026-04-17 - Final Data Sanity Check

## Scope

- Updated `data/combined/train.json`
- Updated `data/dpo_pairs.json`
- Kept `data/combined/eval.json` unchanged

## Goal

Run a strict final sanity pass to remove misleading supervision signals before training.

## Checks Enforced

1. Label correctness
- `expected_vulnerable = false` requires safe parameterized output.
- `expected_vulnerable = true` requires real vulnerable behavior.

2. SQL correctness
- For outputs using parameter tuples, placeholder count must match argument count.
- SQL structure must remain executable in principle (no obviously broken binding shape).

3. Semantic consistency
- Output must match sample schema context (table/column alignment with prompt/input).

4. DPO constraints
- `chosen` must be strictly better (safe/correct) than `rejected`.
- `chosen` and `rejected` must solve the same task and schema.

## Actions Taken

- Fixed trivial issues when deterministic repair was not needed by deleting uncertain samples (per policy).
- Removed samples failing strict checks instead of guessing repairs.
- No new samples added.
- No new prompt/task pattern introduced.
- No broad rewrite performed in this pass.

## Resulting Size Changes

- `train.json`: `4400 -> 3733` (removed `667`)
- `dpo_pairs.json`: `4400 -> 3449` (removed `951`)

## Expected Impact

- Reduces contradictory labels and ambiguous supervision.
- Improves trustworthiness of safety preference signals in DPO.
- Lowers risk of training on semantically misaligned or mechanically invalid samples.
