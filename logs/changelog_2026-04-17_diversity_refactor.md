# Changelog - 2026-04-17 - Dataset Diversity Refactor

## Scope

- Updated `data/combined/train.json`
- Updated `data/dpo_pairs.json`
- Kept `data/combined/eval.json` unchanged

## What Was Improved

### 1) Reduced template repetition while preserving domain

- Reworked answer code structures without changing the SQL/security domain.
- Preserved original task intent and metadata (task type, vulnerability type, expected vulnerability).
- Added controlled structural variety across samples:
  - single-line SQL and multi-line SQL variants
  - helper-function based query construction
  - lightweight optional branching (e.g., static extra condition flags)

### 2) Kept correctness/safety behavior consistent with labels

- For safe targets, retained parameterized binding behavior.
- For vulnerable targets, retained intentional unsafe patterns aligned to attack type.
- Avoided introducing new task templates or new sample records.

### 3) DPO pair diversity and consistency

- Updated `chosen/rejected` code forms to be less repetitive while staying on the same task/schema.
- Maintained pair-level contrast on quality (safe vs unsafe) with same table/column context.

### 4) Light phrase/comment cleanup

- Removed repeated boilerplate artifacts and non-informative suffixes carried from earlier generations.
- Kept only code-relevant content.

## Constraints Compliance

- No new samples added.
- No task domain shift.
- No task-type changes.
- Difficulty kept roughly equivalent (structural variation only, not complexity escalation).

## Expected Impact

- Lower short-pattern memorization risk caused by near-identical outputs.
- Better robustness to minor code-shape variation at training time.
- Improved instruction-following stability due to richer but still controlled output forms.
