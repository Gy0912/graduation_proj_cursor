# Changelog - 2026-04-17 - Merge New Eval Data Into Existing Path

## Scope

- Updated `data/combined/eval.json`
- Updated `README.md`
- Updated `dataset/README.md`
- Removed:
  - `data/combined/eval_generalization_300.json`
  - `data/combined/eval_generalization_600_augmented.json`
  - `data/combined/eval_generalization_660_expanded_unique.json`

## Objective

Integrate newly generated evaluation data into the existing project dataset path without changing loader paths or project structure.

## What Was Done

### 1) Merged new eval data into existing file

- Source used: previously generated high-difficulty eval set (latest merged source).
- Target: `data/combined/eval.json` (existing eval file used by pipeline).
- Conversion to existing schema:
  - kept JSONL line format
  - each entry contains `id`, `prompt`, `meta`
- Deduplication:
  - prompt-based dedup against existing eval entries
  - only non-duplicate prompts were appended

### 2) Removed temporary/new eval JSON files

- Deleted all generated standalone eval files after merge.
- No new dataset loading path introduced.

### 3) Documentation cleanup

- Removed references to standalone generated eval files.
- Updated docs to indicate unified eval data now lives in `data/combined/eval.json`.

## Validation

- `data/combined/eval.json` parse check passed.
- Entry count: `274`
- ID integrity:
  - unique IDs: `274`
  - range: `0..273`
- Schema check passed (`prompt` and `meta` present on all rows).
- Confirmed no remaining `eval_generalization_*.json` files under `data/combined/`.

## Impact

- Project runtime paths remain unchanged.
- Evaluation pipeline continues to read `data/combined/eval.json` exactly as before.
- Newly generated eval content is integrated without structural drift.
