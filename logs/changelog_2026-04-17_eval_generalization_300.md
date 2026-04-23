# Changelog - 2026-04-17 - Eval Generalization Set (300)

## Scope

- Added `data/combined/eval_generalization_300.json`
- Updated `README.md`
- Updated `dataset/README.md`

## What Was Added

### 1) New evaluation-only dataset

- Created a new evaluation dataset with exactly **300** samples:
  - `Level 1`: 100
  - `Level 2`: 100
  - `Level 3`: 100
- Sample schema:
  - `instruction`
  - `input_code`
  - `expected_output`
  - plus `id` and `difficulty_level` for traceability and filtering.

### 2) Difficulty design and reasoning coverage

- `Level 1` (ID): baseline-realistic but non-trivial snippets with small branching and filtering context.
- `Level 2` (Near-OOD): dynamic clause assembly, optional branches, helper-driven filtering, multi-condition control flow.
- `Level 3` (Hard-OOD): JOIN/subquery/ORM misuse and misleading safe-looking vulnerable constructions that require multi-step reasoning.

### 3) Quality constraints enforced

- Dataset is **evaluation-only** and does not alter training files.
- Maintained SQL/security domain focus (Python SQL injection mitigation).
- Avoided trivial one-line pattern replacement by including realistic production-like structure and conditional logic.
- Kept dataset size controlled (no size explosion).

## Documentation Updates

- `README.md`: added a section to verify `eval_generalization_300.json` size and difficulty distribution via PowerShell-friendly command.
- `dataset/README.md`: added this file to main output list with clear non-training usage note.

## Expected Impact

- Improves ability to evaluate model generalization and reasoning beyond in-distribution templates.
- Provides a clearer difficulty ladder for failure analysis (ID / Near-OOD / Hard-OOD).
