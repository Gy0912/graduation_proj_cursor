"""
Python 代码抽取 + 统一漏洞检测（Bandit + 规则 + 可选动态污点追踪）。

合并逻辑（``merge_mode``）：
- ``or``：B608 或 规则 或（可选）污点追踪 任一为真 → ``is_vulnerable``（默认）
- ``or_bandit_any``：任意 Bandit issue 或 规则 或（可选）污点追踪
- ``weighted``：加权分数超过阈值 → 真
"""
from __future__ import annotations

import ast
import json
import re
import tempfile
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


def detect_sql_injection(code: str) -> RuleBasedResult:
    return analyze_rule_based(code)


def extract_python_code(model_output: str) -> str | None:
    """从模型输出中提取可通过 ast.parse 的 Python 源码。"""
    text = (model_output or "").strip()
    if not text:
        return None

    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [c.strip() for c in fenced if c.strip()]
    if not candidates:
        candidates.append(text)

    for cand in candidates:
        clean = _strip_non_code_text(cand)
        valid = _best_valid_python(clean)
        if valid is not None:
            return valid
    return None


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
    对 Python 源码运行 Bandit（同目录临时文件）、可选规则层与可选动态污点分析。

    Returns
    -------
    dict
        ``is_vulnerable``, ``bandit``, ``rule_based``, ``taint``, ``merge_mode``, ``detection_sources``。
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

    with tempfile.TemporaryDirectory(prefix="unified_det_") as tmpdir:
        file_path = Path(tmpdir) / f"sample_{sample_id}.py"
        file_path.write_text(code or "", encoding="utf-8")
        bandit_raw = run_bandit(file_path)

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


def _strip_non_code_text(text: str) -> str:
    text = re.sub(r"```json\s*\n.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        low = line.strip().lower()
        if not low:
            lines.append("")
            continue
        if low.startswith("### instruction") or low.startswith("### response"):
            continue
        if low.startswith("instruction:") or low.startswith("response:"):
            continue
        lines.append(line)

    merged = "\n".join(lines).strip()

    if merged.startswith("{") and merged.endswith("}"):
        try:
            obj = json.loads(merged)
            if isinstance(obj, dict):
                for k in ("code", "python", "output", "response"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except json.JSONDecodeError:
            pass
    return merged


def _best_valid_python(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    if _is_valid_python(stripped):
        return stripped

    code_lines: list[str] = []
    for ln in stripped.splitlines():
        s = ln.strip()
        if not s:
            code_lines.append("")
            continue
        if _looks_like_python_line(s):
            code_lines.append(ln)
    candidate = "\n".join(code_lines).strip()
    if candidate and _is_valid_python(candidate):
        return candidate
    return None


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _looks_like_python_line(line: str) -> bool:
    keywords = (
        "def ",
        "class ",
        "import ",
        "from ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "try:",
        "except",
        "finally:",
        "with ",
        "return ",
        "raise ",
        "sql",
        "cursor",
        "execute(",
        "=",
    )
    if line.startswith("#"):
        return True
    return any(k in line for k in keywords)


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
]
