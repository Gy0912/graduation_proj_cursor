"""数据集共享工具：代码抽取、语法校验、SQL 注入模式检测。

本模块被 ``dataset/generate_expanded_dataset.py``、``training/sft_preprocess.py`` 与
``scripts/check_adversarial_dataset.py`` 共同引用。当前管线已改为 code-only，不再
依赖结构化 marker 输出契约；校验主线为「能抽出 Python + ``ast.parse`` 通过」。

------------------------------------------------------------
为什么要有这个模块
------------------------------------------------------------

旧版 ``generate_expanded_dataset.py`` 的 ambiguous 分支会把**真实可执行的脆弱 SQL
代码**写进 ``output`` 字段并交给 SFT 作为目标序列。一旦模型最小化 next-token
loss，它就是在被**手把手教会**生成 SQL 注入。这是典型的训练数据污染。

用户要求不删除这些样本，而是把 ``output`` 替换成三段式的**拒绝 + 解释 + 安全
替代**文本（对抗训练 target），让模型学到：

1. 能识别不安全指令；
2. 拒绝产出不安全实现；
3. 主动给出参数化查询的安全替代。

为保证「所有写入 training target 的代码片段都是参数化查询」这条硬不变式可以被
机器检验，本模块提供：

- ``extract_code_only_completion(...)``：从输出文本抽取可训练 Python（fence 或整段）；
- ``contains_vulnerable_sql_pattern(...)``：基于 ``detection/rule_based.py``
  的模式集的独立副本（故意 vendoring），让本模块在数据管线里不依赖检测侧；
- ``check_adversarial_dataset(...)``：汇总整份数据集的语法合规率 + 逐条错误明细。

"""
from __future__ import annotations

import re
import ast
from dataclasses import dataclass, field
from typing import Any, Iterable
_SAFE_PYTHON_FENCE_RE = re.compile(
    r"```[ \t]*python[ \t]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def _collapse_identical_halves(code: str) -> str:
    """若整段为同一 Python 片段重复 2/4/… 遍，折叠为单份。"""
    c = code.rstrip()
    while len(c) >= 2 and len(c) % 2 == 0:
        half = len(c) // 2
        if c[:half] != c[half:]:
            break
        c = c[:half]
    return c


def _first_fenced_python_or_whole(text: str) -> str | None:
    """取第一段 ```python ... ```；若无 fence 且整体可 parse，则返回整体。"""
    t = text.strip()
    if not t:
        return None
    m = _SAFE_PYTHON_FENCE_RE.search(t)
    if m:
        body = m.group("code")
        return body.strip("\n") if body else None
    try:
        ast.parse(t)
    except SyntaxError:
        return None
    return t


def extract_code_only_completion(raw: str) -> str | None:
    """
    将输出文本规范为纯 Python（无 fence），不再依赖任何 marker。
    """
    s = (raw or "").strip()
    if not s:
        return None

    code = _first_fenced_python_or_whole(s)

    if not code:
        return None
    code = _collapse_identical_halves(code.strip())
    return code.strip() or None

# --- 脆弱 SQL 模式（与 detection/rule_based.py 平行的 vendoring 副本） ---
#
# 故意复制而不是 import detection.rule_based：
#  1. 数据管线（dataset/ + training/ + scripts/）应当独立于运行时检测栈；
#  2. 二者都会演化，这里保留本模块的「训练目标安全性」合约，可以在不触发
#     检测链联动修改的情况下独立迭代；
#  3. 一旦 detection 侧新增模式，可复制过来并在 CHANGELOG 中互相引用。

_CHECKED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
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
        "percent_format_sql_assigned",
        re.compile(
            r'["\'][^"\']*%[sd][^"\']*["\']\s*%\s*\(',
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
        "execute_plus_concat",
        re.compile(
            r"execute\s*\(\s*[^)]*\+",
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
)


def contains_vulnerable_sql_pattern(code: str) -> tuple[bool, list[str]]:
    """在 Python 代码里扫描已知的 SQL 注入模式。

    返回 ``(found, matched_names)``；``matched_names`` 方便定位具体哪条规则命中，
    供 CI/logs 打印 violation 详情。
    """
    if not isinstance(code, str) or not code:
        return (False, [])
    matched: list[str] = []
    for name, pat in _CHECKED_PATTERNS:
        if pat.search(code):
            matched.append(name)
    return (len(matched) > 0, matched)


# --- 对整份数据集做扫描的汇总函数 ---


@dataclass
class DatasetCheckReport:
    total_samples: int = 0
    parsed_ok: int = 0
    parse_failed: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def parse_pass_rate(self) -> float:
        if self.total_samples == 0:
            return 100.0
        return 100.0 * self.parsed_ok / self.total_samples

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "parsed_ok": self.parsed_ok,
            "parse_failed": self.parse_failed,
            "parse_pass_rate_pct": round(self.parse_pass_rate, 4),
            "violations": list(self.violations),
        }


def _iter_records(records: Iterable[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any]]]:
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise TypeError(
                f"record #{i} must be a dict (got {type(r).__name__})"
            )
        yield i, r


def check_adversarial_dataset(records: Iterable[dict[str, Any]]) -> DatasetCheckReport:
    """对整份数据集做 code-only 语法校验（``ast.parse(output)``）。"""
    report = DatasetCheckReport()
    for i, r in _iter_records(records):
        report.total_samples += 1
        output = str(r.get("output", "")).strip()
        if not output:
            report.parse_failed += 1
            report.violations.append(
                {
                    "index": i,
                    "id": r.get("id"),
                    "kind": "empty_output",
                }
            )
            continue
        try:
            ast.parse(output)
        except SyntaxError as exc:
            report.parse_failed += 1
            report.violations.append(
                {
                    "index": i,
                    "id": r.get("id"),
                    "kind": "ast_parse_failed",
                    "reason": str(exc),
                }
            )
            continue
        report.parsed_ok += 1
    return report
