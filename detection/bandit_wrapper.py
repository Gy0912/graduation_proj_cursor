from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_bandit(file_path: str | Path) -> dict[str, Any]:
    """
    对单个 Python 文件执行 Bandit，并返回统一结构：
    {
      "has_issue": bool,
      "issues": list[dict]
    }
    """
    file_path = str(file_path)
    cmd = ["bandit", file_path, "-f", "json"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = _safe_parse_bandit_json(proc.stdout)
    raw_issues = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(raw_issues, list):
        raw_issues = []
    issues = [_normalize_issue(x) for x in raw_issues]
    has_issue = len(issues) > 0
    return {
        "has_issue": has_issue,
        # 兼容旧逻辑（不移除）
        "is_vulnerable": has_issue,
        "issues": issues,
    }


def _safe_parse_bandit_json(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {"results": []}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"results": []}
    if not isinstance(parsed, dict):
        return {"results": []}
    return parsed


def _normalize_issue(issue: Any) -> dict[str, Any]:
    """
    不过滤 issue，仅做字段规范化，便于后续稳定统计。
    """
    if not isinstance(issue, dict):
        return {
            "test_id": "UNKNOWN",
            "severity": "UNKNOWN",
            "confidence": "UNKNOWN",
            "text": str(issue),
            "line_number": -1,
        }
    return {
        "test_id": str(issue.get("test_id", "UNKNOWN")),
        "severity": str(issue.get("issue_severity", issue.get("severity", "UNKNOWN"))),
        "confidence": str(issue.get("issue_confidence", issue.get("confidence", "UNKNOWN"))),
        "text": str(issue.get("issue_text", issue.get("text", ""))),
        "line_number": int(issue.get("line_number", -1)) if str(issue.get("line_number", "-1")).lstrip("-").isdigit() else -1,
    }
