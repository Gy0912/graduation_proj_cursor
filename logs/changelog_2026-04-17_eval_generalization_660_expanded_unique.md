# Changelog - 2026-04-17 - Eval Expansion (Quality-First Unique Set)

## Scope

- Added `data/combined/eval_generalization_660_expanded_unique.json`
- Updated `README.md`
- Updated `dataset/README.md`

## Goal

Expand an existing evaluation dataset **without reducing difficulty**, while prioritizing uniqueness over raw quantity.

## What Was Done

### 1) Base and expansion policy

- Base file: `data/combined/eval_generalization_600_augmented.json`
- New file: `data/combined/eval_generalization_660_expanded_unique.json`
- Expansion mode:
  - candidate generation across ID / Near-OOD / Hard-OOD
  - near-duplicate filtering by structural + token-overlap checks
  - retain only samples that introduce sufficiently new reasoning/control-flow structure

### 2) Quality-first filtering

- Strictly rejected candidates that were too close to existing structures.
- Final accepted additions:
  - `Level 1`: +5
  - `Level 2`: +5
  - `Level 3`: +5
- Final size: `615` total
- Final balanced distribution: `205 / 205 / 205`

### 3) Constraint compliance

- No difficulty downgrade.
- No trivial one-line variations accepted as new records.
- No task-domain drift (still SQL injection mitigation in Python).
- New records remain evaluation-oriented and semantically valid as fix targets.

## Why Quantity Is Limited

The filter intentionally favors structural novelty and discards high-similarity candidates, so growth is moderate but quality is higher.

## Expected Impact

- Better evaluation signal for generalization and reasoning.
- Lower risk of benchmark inflation from near-duplicate patterns.
