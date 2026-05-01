"""对抗训练（adversarial SFT）共享工具：3 段式安全响应生成、格式校验、SQL 注入模式检测。

本模块是整条管线里「对抗格式」的**唯一真相源**（single source of truth），被
``dataset/generate_expanded_dataset.py``、``training/sft_preprocess.py`` 以及
``scripts/check_adversarial_dataset.py`` 共同引用，禁止再在上述文件里写死 marker
字符串或重复实现检查逻辑——所有对 ``expected_vulnerable=True`` 样本输出结构的
断言、所有对 Python 输出做「是否含 SQL 注入模式」判断都要走这里的 API。

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

- ``build_secure_response(...)``：按 attack 类别合成 3 段式响应；
- ``ADVERSARIAL_MARKERS``：3 段 marker 的唯一字符串常量；
- ``extract_safe_solution(...)`` / ``assert_adversarial_output_format(...)``：
  把 SAFE SOLUTION 片段从对抗响应里抽出来交给下游检测；
- ``contains_vulnerable_sql_pattern(...)``：基于 ``detection/rule_based.py``
  的模式集的独立副本（故意 vendoring），让本模块在数据管线里不依赖检测侧；
- ``check_adversarial_dataset(...)``：汇总整份数据集的对抗样本合规率 + 逐条
  输出扫描，提供 FAIL FAST 的校验入口。

------------------------------------------------------------
安全 SOLUTION 的硬契约（checker 会强制）
------------------------------------------------------------

``build_secure_response`` 合成的 SAFE SOLUTION 代码片段必须：

- 只使用驱动提供的占位符（``%s`` / ``?`` / ``:name``）+ 参数元组/字典；
- 不使用字符串拼接（``"SELECT ..." + x``）；
- 不使用 f-string（``f"SELECT ... {x}"``）；
- 不使用 ``%`` / ``.format()`` 格式化 SQL；
- 不走 ``sqlalchemy.text(f"..."`` / ``text("...{x}...")`` 这类 ORM 误用。
"""
from __future__ import annotations

import random
import re
import ast
from dataclasses import dataclass, field
from typing import Any, Iterable


# --- 唯一的 marker 常量（禁止在其它地方重复） ---
MARKER_WARNING = "[SECURITY WARNING]"
MARKER_EXPLANATION = "[EXPLANATION]"
MARKER_SAFE = "[SAFE SOLUTION]"
ADVERSARIAL_MARKERS: tuple[str, ...] = (MARKER_WARNING, MARKER_EXPLANATION, MARKER_SAFE)


# --- 攻击族 → 警告文本（短句，不含可执行代码样例） ---
_WARNING_BY_ATTACK: dict[str, str] = {
    "string_concat": (
        "The request expects me to assemble an SQL query by concatenating "
        "untrusted input into the SQL text, which is the canonical SQL "
        "injection pattern and is unsafe to implement as written."
    ),
    "fstring": (
        "The request asks me to interpolate an untrusted value into the SQL "
        "via an f-string, which is equivalent to string concatenation and "
        "would inject attacker-controlled syntax into the query."
    ),
    "format_string": (
        "The request wants the SQL built via .format() or percent "
        "formatting on the query text, which inlines untrusted data into "
        "the statement and is indistinguishable from a concatenation-based "
        "injection."
    ),
    "fake_sanitization": (
        "The request relies on ad-hoc character blacklisting (stripping "
        "quotes, semicolons, or comment markers) to defend a concatenated "
        "SQL string; such filters are routinely bypassed and do not make "
        "the query injection-safe."
    ),
    "orm_misuse": (
        "The request uses sqlalchemy.text() with a user-controlled fragment "
        "baked into the raw SQL, which bypasses the ORM's parameter "
        "binding and re-introduces SQL injection."
    ),
    "parameterized_query": (
        "The request asks for pseudo-parameterized code (missing parameter "
        "tuple, wrong arity, or percent-formatted SQL literal) where the "
        "placeholder only looks like a bind; it is still an injection."
    ),
    "indirect_injection": (
        "The request hides string concatenation behind helper functions "
        "and misleading names so the untrusted value still reaches "
        "cursor.execute concatenated into the SQL text."
    ),
}


# --- 攻击族 → 解释文本（2~3 句，避免再复述脆弱 SQL） ---
_EXPLANATION_BY_ATTACK: dict[str, str] = {
    "string_concat": (
        "When a character the attacker can place in the value ends up "
        "inside the concatenated literal, the database parses the "
        "attacker's tokens as SQL syntax, so authentication, row filtering, "
        "and tenant isolation can all be subverted. "
        "Parameterized queries (driver placeholders plus a parameter tuple "
        "or dict) keep the SQL grammar static and pass the untrusted value "
        "as opaque data the database will never interpret as SQL."
    ),
    "fstring": (
        "f-string interpolation happens in Python before the driver sees "
        "the query, so the resulting SQL is fully substituted and any "
        "injected quote or stacked statement changes the parse tree. "
        "Using a driver placeholder with a parameter tuple keeps the "
        "SQL text static and binds the untrusted value at execute time."
    ),
    "format_string": (
        "Dot-format and percent-formatting on the SQL string mutate the query "
        "text in Python, so the database receives a single attacker-"
        "influenced statement. The correct form is to leave the SQL "
        "string unchanged and pass untrusted values as bound parameters "
        "(tuple or dict) so the driver can escape them at bind time."
    ),
    "fake_sanitization": (
        "Blacklists can always be bypassed (alternative encodings, nested "
        "comments, or homoglyphs) and cannot prove that the resulting SQL "
        "string is safe. The only reliable defense is to keep the SQL "
        "text static and pass untrusted values as bound parameters so "
        "they can never be re-interpreted as SQL syntax."
    ),
    "orm_misuse": (
        "Wrapping a concatenated string in the SQLAlchemy raw text API binds "
        "nothing: the untrusted value was already merged into the raw "
        "SQL before SQLAlchemy saw it. The injection-safe form uses a "
        "named placeholder inside the statement and passes the value through "
        "the ORM parameter dictionary."
    ),
    "parameterized_query": (
        "Without a matching parameter tuple or dict the placeholder is "
        "never bound, and formatting the SQL literal via percent substitution "
        "or dot-format before execution turns the placeholder into an attacker-"
        "controllable token. A correct parameterized call keeps the SQL "
        "string static and passes parameters whose length matches the "
        "placeholders."
    ),
    "indirect_injection": (
        "Multi-function data flow and misleading names do not change the "
        "fact that the untrusted value reaches the cursor concatenated "
        "into the SQL text. The only robust defense is to make the SQL "
        "static at the call site and let the driver or ORM bind the "
        "value as a parameter."
    ),
}


# --- Canonical safe SOLUTION 模板：严格参数化 ---


def _safe_pymysql_solution(table: str, column: str) -> str:
    return (
        "import pymysql\n"
        "from pymysql.cursors import DictCursor\n"
        "\n"
        "\n"
        "def query_row(conn: pymysql.connections.Connection, value: str):\n"
        f'    sql = "SELECT * FROM {table} WHERE {column} = %s"\n'
        "    with conn.cursor(DictCursor) as cur:\n"
        "        cur.execute(sql, (value,))\n"
        "        return cur.fetchall()\n"
    )


def _safe_sqlite_solution(table: str, column: str) -> str:
    return (
        "import sqlite3\n"
        "\n"
        "\n"
        "def query_row(conn: sqlite3.Connection, value: str):\n"
        f'    sql = "SELECT * FROM {table} WHERE {column} = ?"\n'
        "    cur = conn.cursor()\n"
        "    cur.execute(sql, (value,))\n"
        "    return cur.fetchall()\n"
    )


def _safe_sqlalchemy_solution(table: str, column: str) -> str:
    return (
        "from sqlalchemy import text\n"
        "from sqlalchemy.orm import Session\n"
        "\n"
        "\n"
        "def query_row(session: Session, value: str):\n"
        f'    stmt = text("SELECT * FROM {table} WHERE {column} = :v")\n'
        '    return session.execute(stmt, {"v": value}).fetchall()\n'
    )


def _choose_safe_solution(attack: str, table: str, column: str, rng: random.Random) -> str:
    """按攻击类型选择合适的安全范例（强制参数化 + 静态 SQL 文本）。"""
    if attack == "orm_misuse":
        return _safe_sqlalchemy_solution(table, column)
    if rng.random() < 0.5:
        return _safe_pymysql_solution(table, column)
    return _safe_sqlite_solution(table, column)


def build_secure_response(
    vulnerable_code: str,
    table: str,
    column: str,
    *,
    attack: str = "string_concat",
    rng: random.Random | None = None,
) -> str:
    """构建 3 段式对抗响应。

    - ``vulnerable_code``: 原来会被写进 ``output`` 的脆弱 SQL 代码；**不会**被
      回显到响应里（避免把脆弱模式重新塞回训练目标）；它的存在是作为语义契约
      占位——调用方在替换 output 时明确交接了「它是什么」。
    - ``table`` / ``column``: 用于生成 SAFE SOLUTION 代码里的静态 SQL 文本；
      这两个值来自数据生成器，受控且可枚举，**不是用户输入**。
    - ``attack``: 攻击族（fstring / fake_sanitization / ...）；用来挑选配套的
      警告/解释文本与安全范例，默认 ``string_concat``。
    - ``rng``: 可选 ``random.Random`` 实例，用于在多条等价的安全范例里选取。

    返回值形如::

        [SECURITY WARNING]
        <一句说明不安全原因>

        [EXPLANATION]
        <2~3 句解释 SQL 注入风险>

        [SAFE SOLUTION]
        <严格参数化的 Python 代码>

    保证（由 ``scripts/check_adversarial_dataset.py`` 机器校验）：

    - 返回字符串必须同时包含 3 段 marker；
    - 紧接 ``[SAFE SOLUTION]`` 的代码块必须使用占位符参数化，
      **不**含字符串拼接 / f-string / ``.format()`` / ``%`` 格式化。
    """
    if rng is None:
        rng = random.Random()
    if not isinstance(vulnerable_code, str):
        raise TypeError(
            f"vulnerable_code must be str (got {type(vulnerable_code).__name__}); "
            "caller is expected to pass the original ambiguous-branch output "
            "as a semantic handoff."
        )
    warning = _WARNING_BY_ATTACK.get(attack, _WARNING_BY_ATTACK["string_concat"])
    explanation = _EXPLANATION_BY_ATTACK.get(attack, _EXPLANATION_BY_ATTACK["string_concat"])
    safe_solution = _choose_safe_solution(attack, table, column, rng).rstrip() + "\n"

    return (
        f"{MARKER_WARNING}\n"
        f"{warning}\n"
        "\n"
        f"{MARKER_EXPLANATION}\n"
        f"{explanation}\n"
        "\n"
        f"{MARKER_SAFE}\n"
        "```python\n"
        f"{safe_solution}"
        "```\n"
    )


def build_safe_response(safe_code: str) -> str:
    """Wrap a safe reference implementation in the same code-only SAFE SOLUTION contract."""
    if not isinstance(safe_code, str) or not safe_code.strip():
        raise ValueError("safe_code must be a non-empty Python string")
    code = safe_code.strip() + "\n"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"safe_code is not valid Python: {exc}") from exc
    return (
        f"{MARKER_WARNING}\n"
        "No unsafe SQL construction is needed for this request.\n"
        "\n"
        f"{MARKER_EXPLANATION}\n"
        "The query text remains static and untrusted values are bound as data by the database driver or ORM.\n"
        "\n"
        f"{MARKER_SAFE}\n"
        "```python\n"
        f"{code}"
        "```\n"
    )


# --- SAFE SOLUTION 抽取 + 完整格式校验 ---

_SAFE_BLOCK_RE = re.compile(
    r"\[SAFE SOLUTION\]\s*(?P<section>.*?)(?:\Z|\n\s*\[[A-Z ]+\])",
    re.DOTALL,
)
_SAFE_PYTHON_FENCE_RE = re.compile(
    r"```[ \t]*python[ \t]*\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_EXPLANATION_SECTION_RE = re.compile(
    r"\[EXPLANATION\]\s*(?P<section>.*?)(?:\Z|\n\s*\[[A-Z ]+\])",
    re.DOTALL,
)
_EXPLANATION_CODELIKE_RE = re.compile(
    r"```|(?:\b(?:execute|executemany|format|text|cursor|fetchall)\s*\()|(?:^\s*[A-Za-z_][A-Za-z0-9_]*\s*=)",
    re.MULTILINE,
)


def extract_safe_solution(output: str) -> str | None:
    """从对抗响应里抽出 SAFE SOLUTION 代码块，失败返回 None。"""
    if not isinstance(output, str) or MARKER_SAFE not in output:
        return None
    m = _SAFE_BLOCK_RE.search(output)
    if not m:
        return None
    fence = _SAFE_PYTHON_FENCE_RE.search(m.group("section"))
    if not fence:
        return None
    code = fence.group("code")
    return code.strip("\n") if code else None


def _explanation_is_prose_only(output: str) -> bool:
    m = _EXPLANATION_SECTION_RE.search(output)
    if not m:
        return False
    return _EXPLANATION_CODELIKE_RE.search(m.group("section")) is None


def assert_adversarial_output_format(output: str) -> None:
    """``expected_vulnerable=True`` 样本的 output 必须同时含 3 段 marker，否则 raise。"""
    if not isinstance(output, str):
        raise TypeError(f"output must be str (got {type(output).__name__})")
    missing = [m for m in ADVERSARIAL_MARKERS if m not in output]
    if missing:
        raise ValueError(
            f"Adversarial output missing markers {missing}; "
            f"expected all of {list(ADVERSARIAL_MARKERS)}. "
            f"first 160 chars: {output[:160]!r}"
        )
    safe_code = extract_safe_solution(output)
    if safe_code is None:
        raise ValueError(
            "Adversarial output has all markers but SAFE SOLUTION has no fenced python code block"
        )
    try:
        ast.parse(safe_code)
    except SyntaxError as exc:
        raise ValueError(
            f"Adversarial SAFE SOLUTION code is not valid Python: {exc}"
        ) from exc
    if not _explanation_is_prose_only(output):
        raise ValueError(
            "Adversarial EXPLANATION must be prose-only: no fences, calls, or assignments"
        )


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
    adversarial_samples: int = 0
    format_compliant: int = 0
    safe_solution_clean: int = 0
    negative_samples: int = 0
    negative_clean: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def format_compliance_rate(self) -> float:
        if self.adversarial_samples == 0:
            return 100.0
        return 100.0 * self.format_compliant / self.adversarial_samples

    @property
    def safe_solution_clean_rate(self) -> float:
        if self.adversarial_samples == 0:
            return 100.0
        return 100.0 * self.safe_solution_clean / self.adversarial_samples

    @property
    def negative_clean_rate(self) -> float:
        if self.negative_samples == 0:
            return 100.0
        return 100.0 * self.negative_clean / self.negative_samples

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "adversarial_samples": self.adversarial_samples,
            "format_compliant": self.format_compliant,
            "format_compliance_rate_pct": round(self.format_compliance_rate, 4),
            "safe_solution_clean": self.safe_solution_clean,
            "safe_solution_clean_rate_pct": round(self.safe_solution_clean_rate, 4),
            "negative_samples": self.negative_samples,
            "negative_clean": self.negative_clean,
            "negative_clean_rate_pct": round(self.negative_clean_rate, 4),
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
    """对整份数据集做对抗样本 + 脆弱 SQL 扫描。

    契约：
      - 每条 ``expected_vulnerable=True`` 样本的 ``output`` 必须含 3 段 marker；
      - 上述样本的 SAFE SOLUTION 代码必须通过 ``contains_vulnerable_sql_pattern``
        为 ``False`` 的扫描（无拼接 / f-string / % 格式 / .format / ORM 误用）；
      - 每条 ``expected_vulnerable=False`` 样本的 ``output`` 整体也必须扫描通过
        （作为上游回归：安全分支本就不该含脆弱模式）。

    任何违规都会被收集到 ``report.violations``；函数**本身不 raise**，交由调用方
    （``check_adversarial_dataset.py`` CLI 或训练侧 pre-flight）决定是否 FAIL FAST。
    """
    report = DatasetCheckReport()
    for i, r in _iter_records(records):
        report.total_samples += 1
        if "expected_vulnerable" not in r:
            report.violations.append(
                {
                    "index": i,
                    "id": r.get("id"),
                    "kind": "missing_expected_vulnerable",
                }
            )
            continue
        ev = r["expected_vulnerable"]
        if not isinstance(ev, bool):
            report.violations.append(
                {
                    "index": i,
                    "id": r.get("id"),
                    "kind": "non_bool_expected_vulnerable",
                    "got": type(ev).__name__,
                }
            )
            continue

        output = str(r.get("output", ""))

        if ev:
            report.adversarial_samples += 1
            missing = [m for m in ADVERSARIAL_MARKERS if m not in output]
            if missing:
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "missing_marker",
                        "missing": missing,
                    }
                )
                continue
            report.format_compliant += 1
            try:
                assert_adversarial_output_format(output)
            except (TypeError, ValueError) as exc:
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "invalid_adversarial_contract",
                        "reason": str(exc),
                    }
                )
                continue

            safe_code = extract_safe_solution(output)
            if safe_code is None or not safe_code.strip():
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "empty_safe_solution",
                    }
                )
                continue
            bad, matches = contains_vulnerable_sql_pattern(safe_code)
            if bad:
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "vulnerable_safe_solution",
                        "matches": matches,
                    }
                )
                continue
            report.safe_solution_clean += 1
        else:
            report.negative_samples += 1
            missing = [m for m in ADVERSARIAL_MARKERS if m not in output]
            if missing:
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "missing_marker",
                        "missing": missing,
                    }
                )
                continue
            try:
                assert_adversarial_output_format(output)
            except (TypeError, ValueError) as exc:
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "invalid_output_contract",
                        "reason": str(exc),
                    }
                )
                continue
            safe_code = extract_safe_solution(output)
            bad, matches = contains_vulnerable_sql_pattern(safe_code or "")
            if bad:
                report.violations.append(
                    {
                        "index": i,
                        "id": r.get("id"),
                        "kind": "vulnerable_negative_output",
                        "matches": matches,
                    }
                )
                continue
            report.negative_clean += 1
    return report
