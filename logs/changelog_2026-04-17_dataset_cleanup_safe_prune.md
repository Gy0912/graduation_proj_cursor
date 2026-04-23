# Changelog - 2026-04-17 - Dataset Cleanup (Safe Prune)

## Scope

- Deleted:
  - `dataset/examples/dpo_example.jsonl`
  - `dataset/examples/eval_prompts_example.jsonl`
  - `dataset/examples/sft_example.jsonl`
  - `dataset/eval_prompts.jsonl`
  - `data/dpo_pairs.jsonl`
- Updated:
  - `README.md`
  - `dataset/README.md`
  - `PROJECT_STRUCTURE.md`

## Why These Files Were Removed

- `dataset/examples/*.jsonl` are demonstration samples and are not referenced by runtime training/evaluation code.
- `dataset/eval_prompts.jsonl` is a legacy generated artifact; current config and runtime use `data/combined/eval.json`.
- `data/dpo_pairs.jsonl` is a compatibility duplicate; current DPO training reads `data/dpo_pairs.json`.

## Safety Checks

- Performed repository-wide reference search before deletion.
- Confirmed no core script imports these files as runtime dependency.
- Kept potentially useful compatibility datasets under `data/generation/`, `data/fix/`, `data/schema/`, `data/samples/`, `data/*_expanded.json` to avoid accidental workflow breakage.

## Impact

- Project remains runnable with unchanged entry points and data loading paths.
- Reduced clutter from unused dataset artifacts.
- Documentation now matches actual on-disk dataset files.
