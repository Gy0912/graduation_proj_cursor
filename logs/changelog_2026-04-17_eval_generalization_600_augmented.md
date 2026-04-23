# Changelog - 2026-04-17 - Eval Dataset Augmentation (600)

## Scope

- Added `data/combined/eval_generalization_600_augmented.json`
- Updated `README.md`
- Updated `dataset/README.md`

## Objective

Expand an existing high-quality evaluation dataset to ~2x size while preserving quality, SQL-safety semantics, and structural diversity.

## What Was Done

### 1) Size expansion with controlled augmentation

- Source dataset: `data/combined/eval_generalization_300.json` (300 samples)
- Output dataset: `data/combined/eval_generalization_600_augmented.json` (600 samples)
- Strategy: for each original sample, add 1 derived variant (`-aug1`) with:
  - extra query conditions (e.g., deleted/active filters)
  - helper function extraction
  - split query-building steps
  - optional-branch edge-case handling (e.g., include_inactive / limit normalization)

### 2) Quality and semantic constraints kept

- No domain shift (still Python SQL injection mitigation).
- No trivial copy-only duplication; each added sample modifies control flow or query assembly.
- Maintained safe expected outputs with parameterized binding.
- Preserved schema alignment (table/column consistency) in augmented pairs.

### 3) Difficulty distribution

- `Level 1`: 200
- `Level 2`: 200
- `Level 3`: 200
- IDs unique across all 600 records.

## Validation

- JSON parsing successful for full file.
- Distribution and uniqueness checks passed.
- Regenerated once after detecting and fixing a schema parsing issue in early augmentation output; final file is corrected.

## Expected Impact

- Higher evaluation coverage without introducing noisy template inflation.
- Better stress testing for generalization across helper-based and branch-driven code structures.
