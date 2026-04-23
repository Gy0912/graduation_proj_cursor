# Changelog - 2026-04-17 - Difficulty Augmentation (3B-Oriented)

## Scope

- Updated `data/combined/train.json`
- Updated `data/dpo_pairs.json`
- Kept `data/combined/eval.json` unchanged

## Objective

Increase sample reasoning depth for a 3B model without changing domain, exploding size, or breaking semantic alignment.

## What Was Changed

### 1) Difficulty-focused structure upgrades in training outputs

- Added moderate complexity patterns while preserving original schema alignment (table/column/driver):
  - multi-condition filtering
  - optional conditional branches (`strict_mode` style)
  - dynamic query assembly paths
  - selective join-like structure in hard safe variants (derived-table join on same table context)
- Kept easy/medium/hard tiers roughly intact by applying richer patterns mainly to harder samples.

### 2) DPO preference refinement (subtle but meaningful contrast)

- Reworked `chosen/rejected` so both look plausible and task-aligned.
- Ensured contrast requires reasoning:
  - `chosen`: parameter binding with structured query logic
  - `rejected`: similarly structured code but still unsafe interpolation/concatenation
- Preserved same-task, same-schema pairing across DPO entries.

### 3) Safety and alignment guards

- Maintained semantic consistency between input schema and outputs.
- Preserved task domain (SQL/security) and task metadata.
- Performed a targeted post-fix for `sqlalchemy + hard + safe` placeholders to keep bindings explicit and coherent.

## What Was Not Changed

- No new records added.
- No dataset-size expansion.
- No domain shift outside SQL/security.
- No intentional noise insertion.

## Expected Impact

- Lower shallow pattern memorization risk.
- Better exposure to realistic query-shape variation.
- Stronger DPO signal quality by making rejected answers plausibly close but still wrong.
