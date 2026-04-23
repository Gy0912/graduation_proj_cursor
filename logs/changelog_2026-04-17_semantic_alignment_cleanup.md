# Changelog - 2026-04-17 - Semantic Alignment Cleanup

## Scope

- Updated `data/combined/train.json`
- Updated `data/dpo_pairs.json`
- Checked `data/combined/eval.json` (no content change required)

## What Was Modified

### 1) Semantic alignment fixes in `train.json`

- Rewrote `output` fields to align with each sample's `input_code` and task context.
- Enforced consistency for:
  - table name
  - column name
  - database driver style (e.g., `pymysql`, `sqlite3`, `psycopg2`, `sqlalchemy`)
  - query target logic described by the prompt/input
- Kept each sample's existing metadata (`id`, `task_type`, `difficulty`, `vulnerability_type`, `expected_vulnerable`) unchanged.
- Preserved dataset size (no sample additions, no sample deletions).

### 2) DPO pair corrections in `dpo_pairs.json`

- Rewrote `chosen` and `rejected` for semantic consistency with the same prompt/input schema.
- Enforced that each pair uses the same table/column target and task context.
- Standardized pair quality contrast:
  - `chosen`: safe/correct parameterized behavior
  - `rejected`: lower-quality unsafe behavior matching the same task and schema
- Removed semantically unrelated pair content (e.g., mismatched table/column/driver between prompt and answer).
- Preserved dataset size (same number of pairs retained).

### 3) Light noise cleanup

- Removed repeated non-informative noise fragments from generated answers (e.g., legacy review tags and ambiguous reference suffixes).
- Kept code concise and task-relevant, without introducing new task templates.

## What Was Not Changed

- No new samples were created.
- No task types were changed.
- No simplification of task intent.
- `data/combined/eval.json` remained unchanged after consistency check (metadata already matched prompt content).

## Why These Changes Were Made

- The previous dataset contained semantic mismatches where outputs did not reflect the prompt/input schema.
- Such mismatches create contradictory supervision signals and can train unstable behavior.
- DPO pairs with schema drift reduce preference-learning quality by comparing answers to different tasks.

## Impact on Training/Data Quality

- Improves label and schema consistency across training samples.
- Reduces contradictory supervision and accidental cross-schema leakage.
- Makes DPO preference signals more valid (quality contrast under the same task).
- Keeps the dataset footprint stable while increasing reliability of fine-tuning signals.
