# Changelog: 2026-05-03 — Code extraction supports raw Python (code-only pipeline)

## Summary

Updated `detection/sql_injection_detector.py::extract_python_code` and `extract_python_code_with_debug` so evaluation no longer requires Markdown `` ```python `` fences. The model may emit bare Python (as in the current code-only inference path); extraction succeeds when the full trimmed text passes `ast.parse`, with fenced blocks and a small line-anchor heuristic as fallbacks.

## What changed

- **Added** `_best_fenced_python` and `_heuristic_python_slice` helpers to keep logic in one place and preserve “last fenced block wins” behavior.
- **Priority (both public functions):** `raw` (entire text `ast.parse`) → `python_fence` (existing regex) → `heuristic` (slice from first `def` / `async def` / `class` / `import` / `from` line) → `None` / `source="none"`.
- **`extract_python_code_with_debug`:** `ExtractionResult.source` now distinguishes `raw`, `python_fence`, `heuristic`, and `none` (replacing the previous situation where failures were labeled `python_fence` even when no fence existed). Empty input uses `none`.
- **Preprocessing:** Existing `### Instruction:` / `### Input:` truncation in `extract_python_code_with_debug`, and `### Response` / `### Instruction` / `### Input` slicing in `extract_python_code`, are unchanged.

## Why

After the refactor to code-only outputs, the extractor still only accepted fenced blocks, so `extracted_candidate` was always `None` and evaluations failed. Raw-first ordering accepts training-aligned completions while keeping backward compatibility for fenced outputs. The heuristic covers short natural-language prefixes before valid top-level statements without reintroducing broad “splice arbitrary prose” risk: the slice must still pass `ast.parse`.

## Impact

- **Eval:** Samples with valid raw Python (and optional leading noise before a top-level `def`/`import`/…) are scored again; invalid extractions should drop sharply when the model follows the code-only contract.
- **No** changes to datasets, SFT/DPO scripts, or evaluator control flow—only the detector extraction module and documentation.

## Docs

- README: new section **Code Extraction Strategy**; top-of-file blockquote cross-reference.
- This changelog file.
