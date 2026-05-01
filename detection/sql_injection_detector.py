"""
Python code extraction + unified vulnerability detection
(Bandit + rules + optional dynamic taint tracking).

Merge modes:
- ``or``: B608 or rules or optional taint marks ``is_vulnerable`` true.
- ``or_bandit_any``: any Bandit issue or rules or optional taint.
- ``weighted``: weighted score over threshold.
"""
from __future__ import annotations

import atexit
import ast
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from detection.bandit_wrapper import run_bandit
from detection.taint_tracker import run_taint_analysis
from detection.rule_based import (
    RuleBasedResult,
    SQLInjectionDetector,
    analyze_rule_based,
)

MergeMode = Literal["or", "or_bandit_any", "weighted"]

_WEIGHTED_THRESHOLD = 0.55
_WEIGHTS = {
    "bandit_b608": 0.95,
    "bandit_other": 0.35,
    "rule_based": 0.85,
    "taint": 0.9,
}

_EXTRACTION_STATS = {
    "safe_solution_hits": 0,
    "valid_extractions": 0,
    "invalid_extractions": 0,
}
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_DETECTOR_TMP_ROOT = _WORKSPACE_ROOT / "outputs" / "tmp_detector"

_PYTHON_FENCE_RE = re.compile(
    r"```[ \t]*python[ \t]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ExtractionResult:
    code: str | None
    candidate: str | None
    reason: str
    source: str | None


def _print_extraction_summary() -> None:
    print(
        "[Extraction Fix] "
        f"SAFE_SOLUTION_hits={_EXTRACTION_STATS['safe_solution_hits']} "
        f"valid_extractions={_EXTRACTION_STATS['valid_extractions']} "
        f"invalid_extractions={_EXTRACTION_STATS['invalid_extractions']}"
    )


atexit.register(_print_extraction_summary)


def detect_sql_injection(code: str) -> RuleBasedResult:
    return analyze_rule_based(code)


def extract_python_code_with_debug(model_output: str) -> ExtractionResult:
    text = model_output or ""
    if not text:
        return ExtractionResult(None, None, "no code found", "python_fence")

    instruction_idx = text.find("### Instruction:")
    if instruction_idx != -1:
        text = text[:instruction_idx]

    input_idx = text.find("### Input:")
    if input_idx != -1:
        text = text[:input_idx]

    code_blocks = [m.group("code").strip() for m in _PYTHON_FENCE_RE.finditer(text) if m.group("code").strip()]
    if not code_blocks:
        return ExtractionResult(None, None, "no code found", "python_fence")

    last_candidate: str | None = None
    for candidate in reversed(code_blocks):
        last_candidate = candidate
        ok, reason = _parse_python(candidate)
        if ok:
            return ExtractionResult(candidate, candidate, "ok", "python_fence")
    return ExtractionResult(None, last_candidate, reason, "python_fence")


def extract_python_code(model_output: str) -> str | None:
    text = model_output or ""
    marker_positions = [
        text.rfind("### Response"),
        text.rfind("### Instruction"),
        text.rfind("### Input"),
    ]
    start = max(marker_positions)
    if start != -1:
        text = text[start:]

    matches = list(
        re.finditer(
            r"```[ \t]*python[ \t]*\r?\n(?P<code>.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return None

    candidate = matches[-1].group("code").strip()
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def _parse_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"AST parse error: {exc.msg} at line {exc.lineno}, column {exc.offset}"
    return True, "ok"


def _bandit_sql_flag(issues: list[dict[str, Any]], *, any_issue: bool) -> tuple[bool, bool]:
    if not issues:
        return False, False
    b608 = any(
        str(i.get("test_id", "")).upper() == "B608"
        for i in issues
        if isinstance(i, dict)
    )
    if any_issue:
        return b608, True
    return b608, b608


def _merge(
    bandit_layer: bool,
    b608: bool,
    rule_vuln: bool,
    taint_vuln: bool,
    mode: MergeMode,
    *,
    bandit_issues: list[dict[str, Any]],
) -> tuple[bool, str]:
    if mode == "or_bandit_any":
        b_any = len(bandit_issues) > 0
        return b_any or rule_vuln or taint_vuln, "or_bandit_any"
    if mode == "weighted":
        score = 0.0
        if b608:
            score += _WEIGHTS["bandit_b608"]
        elif bandit_issues:
            score += _WEIGHTS["bandit_other"]
        if rule_vuln:
            score += _WEIGHTS["rule_based"]
        if taint_vuln:
            score += _WEIGHTS["taint"]
        return score >= _WEIGHTED_THRESHOLD, "weighted"
    return bandit_layer or rule_vuln or taint_vuln, "or"


def detect_vulnerability(
    code: str,
    *,
    sample_id: int = 0,
    merge_mode: MergeMode = "or",
    enable_rule_based: bool = True,
    enable_taint: bool = False,
    rule_detector: SQLInjectionDetector | None = None,
) -> dict[str, Any]:
    """
    Run Bandit, the rule-based detector, and optional taint analysis on Python
    source code.

    Returns a dictionary containing ``is_vulnerable``, detector sub-results,
    ``merge_mode``, and ``detection_sources``.
    """
    if enable_rule_based:
        det = rule_detector or SQLInjectionDetector()
        rule_dict = det.analyze(code).to_dict()
    else:
        rule_dict = {
            "is_vulnerable": False,
            "violations": [],
            "matched_patterns": [],
            "details": {"disabled": True},
        }

    _DETECTOR_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    file_path = _DETECTOR_TMP_ROOT / f"sample_{sample_id}_{uuid.uuid4().hex}.py"
    try:
        file_path.write_text(code or "", encoding="utf-8")
        bandit_raw = run_bandit(file_path)
    finally:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass

    issues = bandit_raw.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    any_issue = merge_mode == "or_bandit_any"
    b608_hit, bandit_layer = _bandit_sql_flag(issues, any_issue=any_issue)
    rule_vuln = bool(rule_dict.get("is_vulnerable")) if enable_rule_based else False

    if enable_taint:
        taint_raw = run_taint_analysis(code or "")
        taint_dict = {
            "skipped": False,
            "is_vulnerable": bool(taint_raw.get("is_vulnerable")),
            "taint_flows_detected": int(taint_raw.get("taint_flows_detected", 0)),
            "details": list(taint_raw.get("details", [])),
            "error": taint_raw.get("error"),
        }
    else:
        taint_dict = {
            "skipped": True,
            "reason": "enable_taint=False",
            "is_vulnerable": False,
            "taint_flows_detected": 0,
            "details": [],
            "error": None,
        }

    taint_vuln = bool(taint_dict.get("is_vulnerable")) if enable_taint else False

    merged, resolved_merge = _merge(
        bandit_layer,
        b608_hit,
        rule_vuln,
        taint_vuln,
        merge_mode,
        bandit_issues=issues,
    )

    sources: list[str] = []
    if merge_mode == "or_bandit_any":
        if issues:
            sources.append("bandit")
    else:
        if b608_hit:
            sources.append("bandit_b608")
    if rule_vuln:
        sources.append("rule_based")
    if taint_vuln:
        sources.append("taint")

    return {
        "is_vulnerable": merged,
        "merge_mode": resolved_merge,
        "detection_sources": sources,
        "bandit": {
            "has_issue": bool(bandit_raw.get("has_issue")),
            "is_vulnerable": bool(bandit_raw.get("is_vulnerable")),
            "b608_hit": b608_hit,
            "bandit_layer_used": bandit_layer,
            "issues": issues,
        },
        "rule_based": rule_dict,
        "taint": taint_dict,
    }


def detect_vulnerability_json(code: str, **kwargs: Any) -> str:
    return json.dumps(detect_vulnerability(code, **kwargs), ensure_ascii=False, indent=2)



DetectionResult = RuleBasedResult

__all__ = [
    "DetectionResult",
    "MergeMode",
    "RuleBasedResult",
    "SQLInjectionDetector",
    "analyze_rule_based",
    "detect_sql_injection",
    "detect_vulnerability",
    "detect_vulnerability_json",
    "extract_python_code",
    "extract_python_code_with_debug",
]

