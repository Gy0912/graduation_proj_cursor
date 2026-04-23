"""规则层 SQL 注入相关模式检测（轻量、低延迟，可补 Bandit 盲区）。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleBasedResult:
    is_vulnerable: bool
    violations: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_vulnerable": self.is_vulnerable,
            "violations": list(self.violations),
            "matched_patterns": list(self.matched_patterns),
            "details": dict(self.details),
        }


class SQLInjectionDetector:
    """基于正则与轻量启发式的 SQL 字符串动态构造检测。"""

    def __init__(self) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                "fstring_sql",
                re.compile(
                    r"(?:execute|executemany)\s*\(\s*f[\"']",
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "concat_plus_sql",
                re.compile(
                    r'["\']\s*(?:SELECT|INSERT|UPDATE|DELETE)\b[^"\']*["\']\s*\+',
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "format_sql",
                re.compile(
                    r"(?:execute|executemany)\s*\(\s*[\"'][^\"']*\{[^}]+\}[^\"']*[\"']\s*\.format",
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "percent_format_sql",
                re.compile(
                    r"(?:execute|executemany)\s*\(\s*[\"'][^\"']*%[sd][^\"']*[\"']\s*%",
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "sqlalchemy_text_fstring",
                re.compile(
                    r"text\s*\(\s*f[\"']",
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "sqlalchemy_text_format",
                re.compile(
                    r"text\s*\(\s*[\"'][^\"']*\{[^}]+\}",
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "join_build_sql",
                re.compile(
                    r"(?:execute|executemany)\s*\(\s*[\"'][^\"']*[\"']\s*\.join",
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            # NOTE(2026-04-22 七次加固): 旧版本在此还有一条 ``percent_execute_tuple``
            # 规则（正则 ``execute\s*\(\s*["'][^"']*%s``）。它缺少对字符串闭合与
            # 尾随 ``% `` 运算符的锚定，会把 ``execute("... %s", (val,))`` 这种
            # pymysql / psycopg2 的**参数化查询**与真·格式化 ``"..." % val`` 混为
            # 一谈，造成**高危假阳性**（训练/评测全部把正确的安全写法当成注入）。
            # 由于真·格式化形态已由上方 ``percent_format_sql`` 严格规则（``%[sd]``
            # 后显式要求字符串关闭引号 + ``\s*%``）以及 Bandit B608 共同覆盖，
            # 这条冗余且有害的规则已**整条删除**，不留向后兼容。详见
            # ``logs/changelog_2026-04-22_rule_false_positive_fix.md`` 与
            # ``tests/test_rule_false_positive.py``。
        ]

    def analyze(self, code: str) -> RuleBasedResult:
        text = code or ""
        violations: list[str] = []
        matched: list[str] = []

        for name, pat in self._patterns:
            if pat.search(text):
                violations.append(name)
                matched.append(name)

        if self._unsafe_execute_heuristic(text):
            if "unsafe_execute_heuristic" not in violations:
                violations.append("unsafe_execute_heuristic")
                matched.append("unsafe_execute_heuristic")

        is_vulnerable = len(violations) > 0
        return RuleBasedResult(
            is_vulnerable=is_vulnerable,
            violations=violations,
            matched_patterns=matched,
            details={},
        )

    def _unsafe_execute_heuristic(self, text: str) -> bool:
        if re.search(r"execute\s*\(\s*[^)]*\+", text, re.IGNORECASE):
            return True
        if re.search(r"execute\s*\(\s*f[\"']", text, re.IGNORECASE):
            return True
        if re.search(r"executemany\s*\(\s*f[\"']", text, re.IGNORECASE):
            return True
        return False


def analyze_rule_based(code: str) -> RuleBasedResult:
    return SQLInjectionDetector().analyze(code)
