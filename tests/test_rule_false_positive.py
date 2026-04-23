"""规则层 SQL 注入检测的假阳性回归测试。

对应 **2026-04-22 七次加固**：``detection/rule_based.py`` 原有的
``percent_execute_tuple`` 规则
（正则 ``execute\\s*\\(\\s*["\'][^"\']*%s``）会把**驱动占位符**
``cursor.execute("... %s", (val,))`` 当成 ``"..." % val`` 的字符串格式化漏洞
误报。实际上这种形态是 pymysql / psycopg2 参数化查询的标准写法，必须放行。

本测试覆盖两条相互独立的防线：

* ``TestRuleBasedFalsePositive`` —— 仅规则层（``analyze_rule_based``）。
  删除 ``percent_execute_tuple`` 后，所有参数化写法都应被规则层放行，
  而 ``"...%s" % val`` 由保留下来的 ``percent_format_sql`` 负责（其在 ``%s``
  后面还锚定了字符串关闭引号和紧跟的 ``%`` 运算符，所以不会误伤元组参数）。

* ``TestFullPipelineFalsePositive`` —— 端到端合并层（``detect_vulnerability``）。
  按用户原样给出的 UNSAFE 示例（SQL 字面量里再嵌一对单引号包住 ``%s`` 的写法）
  虽然绕过了规则层的单一正则，但 Bandit B608 会在整管线里覆盖住——这是本次
  修复方案选择"直接删除 ``percent_execute_tuple`` 而不是把它写得更严"的理由。

SAFE 契约：
    cursor.execute("SELECT * FROM t WHERE x = %s", (val,))
UNSAFE 契约：
    cursor.execute("SELECT * FROM t WHERE x = '%s'" % val)

运行：``python -m unittest tests.test_rule_false_positive -v``
"""
from __future__ import annotations

import shutil
import unittest

from detection.rule_based import SQLInjectionDetector, analyze_rule_based
from detection.sql_injection_detector import detect_vulnerability


# ---------------------------------------------------------------------------
# 规则层（单模块）
# ---------------------------------------------------------------------------


class TestRuleBasedFalsePositive(unittest.TestCase):
    """仅针对 ``detection.rule_based`` 的正则规则集，不涉及 Bandit / 污点。"""

    def setUp(self) -> None:
        self.det = SQLInjectionDetector()

    # ------------------------------------------------------------------
    # SAFE 参数化查询 —— 必须 NOT 触发任何规则
    # ------------------------------------------------------------------

    def test_safe_percent_s_with_tuple_param_not_flagged(self) -> None:
        """pymysql / psycopg2 风格 ``%s`` + 元组参数 = 参数化查询，不得被命中。"""
        code = 'cursor.execute("SELECT * FROM t WHERE x = %s", (val,))\n'
        r = analyze_rule_based(code)
        self.assertFalse(
            r.is_vulnerable,
            msg=f"误报：参数化查询被判定为 vulnerable；violations={r.violations}",
        )
        self.assertEqual(r.violations, [])
        self.assertEqual(r.matched_patterns, [])

    def test_safe_percent_s_multiple_params_not_flagged(self) -> None:
        code = (
            'cursor.execute(\n'
            '    "INSERT INTO users (name, email) VALUES (%s, %s)",\n'
            '    (name, email),\n'
            ')\n'
        )
        r = analyze_rule_based(code)
        self.assertFalse(r.is_vulnerable, msg=f"violations={r.violations}")

    def test_safe_percent_s_update_delete_not_flagged(self) -> None:
        for stmt in (
            'cursor.execute("UPDATE t SET a = %s WHERE id = %s", (a, i))',
            'cursor.execute("DELETE FROM t WHERE id = %s", (i,))',
        ):
            with self.subTest(stmt=stmt):
                r = analyze_rule_based(stmt)
                self.assertFalse(
                    r.is_vulnerable,
                    msg=f"误报: {stmt!r} → violations={r.violations}",
                )

    def test_safe_qmark_placeholder_not_flagged(self) -> None:
        """sqlite3 的 ``?`` 占位符同样不得触发任何规则。"""
        code = 'cursor.execute("SELECT * FROM t WHERE x = ?", (val,))\n'
        r = analyze_rule_based(code)
        self.assertFalse(r.is_vulnerable, msg=f"violations={r.violations}")

    def test_safe_named_placeholder_not_flagged(self) -> None:
        """SQLAlchemy / psycopg2 命名参数 ``:name`` / ``%(name)s`` 不得触发。"""
        for stmt in (
            'cursor.execute("SELECT * FROM t WHERE x = :val", {"val": v})',
            'cursor.execute("SELECT * FROM t WHERE x = %(val)s", {"val": v})',
        ):
            with self.subTest(stmt=stmt):
                r = analyze_rule_based(stmt)
                self.assertFalse(
                    r.is_vulnerable,
                    msg=f"误报: {stmt!r} → violations={r.violations}",
                )

    def test_safe_executemany_with_tuple_param_not_flagged(self) -> None:
        code = (
            'cursor.executemany(\n'
            '    "INSERT INTO t (a, b) VALUES (%s, %s)",\n'
            '    rows,\n'
            ')\n'
        )
        r = analyze_rule_based(code)
        self.assertFalse(r.is_vulnerable, msg=f"violations={r.violations}")

    # ------------------------------------------------------------------
    # UNSAFE 真·漏洞 —— 必须仍被规则层捕获（防止"修假阳性"误伤真阳性）
    # ------------------------------------------------------------------

    def test_unsafe_percent_format_in_execute_is_flagged(self) -> None:
        """``"..." % val`` 字符串格式化 → 规则 ``percent_format_sql`` 命中。"""
        code = 'cursor.execute("SELECT * FROM t WHERE x = %s" % val)\n'
        r = analyze_rule_based(code)
        self.assertTrue(r.is_vulnerable, msg=f"漏报: violations={r.violations}")
        self.assertIn("percent_format_sql", r.violations)

    def test_unsafe_percent_format_tuple_args_in_execute_is_flagged(self) -> None:
        """``"..." % (a, b)`` 形式依然是字符串格式化、非参数化。"""
        code = 'cursor.execute("SELECT * FROM t WHERE a=%s AND b=%s" % (a, b))\n'
        r = analyze_rule_based(code)
        self.assertTrue(r.is_vulnerable, msg=f"漏报: violations={r.violations}")
        self.assertIn("percent_format_sql", r.violations)

    def test_unsafe_fstring_execute_is_flagged(self) -> None:
        code = 'cursor.execute(f"SELECT * FROM t WHERE x = {val}")\n'
        r = analyze_rule_based(code)
        self.assertTrue(r.is_vulnerable)
        self.assertIn("fstring_sql", r.violations)

    def test_unsafe_plus_concat_execute_is_flagged(self) -> None:
        code = 'cursor.execute("SELECT * FROM t WHERE x = " + val)\n'
        r = analyze_rule_based(code)
        self.assertTrue(r.is_vulnerable)
        self.assertTrue(
            {"concat_plus_sql", "unsafe_execute_heuristic"} & set(r.violations),
            msg=f"期望命中 concat_plus_sql / unsafe_execute_heuristic，实际 {r.violations}",
        )

    def test_unsafe_str_format_execute_is_flagged(self) -> None:
        code = 'cursor.execute("SELECT * FROM t WHERE x = {0}".format(val))\n'
        r = analyze_rule_based(code)
        self.assertTrue(r.is_vulnerable)
        self.assertIn("format_sql", r.violations)

    # ------------------------------------------------------------------
    # 回归：``percent_execute_tuple`` 规则本身必须已被删除
    # ------------------------------------------------------------------

    def test_percent_execute_tuple_rule_removed(self) -> None:
        """2026-04-22 修复后，规则名 ``percent_execute_tuple`` 不再存在。"""
        rule_names = {name for name, _ in self.det._patterns}
        self.assertNotIn(
            "percent_execute_tuple",
            rule_names,
            msg=(
                "percent_execute_tuple 规则未被移除。该规则正则 "
                "`execute\\s*\\(\\s*[\"'][^\"']*%s` 无法区分参数化查询与 `%` 格式化，"
                "会把 `execute(\"...%s\", (v,))` 误报为漏洞。"
            ),
        )


# ---------------------------------------------------------------------------
# 端到端合并层（Bandit B608 兜底用户原样示例）
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    shutil.which("bandit") is not None,
    "bandit CLI not on PATH; skipping full-pipeline checks",
)
class TestFullPipelineFalsePositive(unittest.TestCase):
    """``detect_vulnerability`` = Bandit + 规则层（默认 ``or`` 合并模式）。

    这里保证用户在 PROBLEM 里给出的**原样两条**（SAFE / UNSAFE）在整管线下
    的分类与 ground truth 一致——即使规则层对嵌套单引号 ``'%s'`` 存在覆盖
    盲区，Bandit B608 会兜住。
    """

    def test_safe_parameterized_not_flagged_by_pipeline(self) -> None:
        code = (
            "import sqlite3\n"
            'conn = sqlite3.connect(":memory:")\n'
            "cursor = conn.cursor()\n"
            "val = 'alice'\n"
            'cursor.execute("SELECT * FROM t WHERE x = %s", (val,))\n'
        )
        r = detect_vulnerability(code)
        self.assertFalse(
            r["is_vulnerable"],
            msg=(
                "整管线对参数化查询 `execute(\"...%s\", (val,))` 产生了误报。"
                f" bandit={r['bandit']}, rule_based={r['rule_based']}"
            ),
        )
        self.assertEqual(r["rule_based"]["violations"], [])
        self.assertFalse(r["bandit"]["b608_hit"])

    def test_unsafe_percent_format_flagged_by_pipeline_exact_user_example(
        self,
    ) -> None:
        """用户 PROBLEM 段原样给出的 UNSAFE（SQL 里嵌 ``'%s'`` + `` % val``）。

        规则层由于 ``[^"']*`` 无法跨越嵌套的单引号，不会命中 ``percent_format_sql``；
        但 Bandit B608 会命中，合并后 ``is_vulnerable`` 仍然为 True。
        """
        code = (
            "import sqlite3\n"
            'conn = sqlite3.connect(":memory:")\n'
            "cursor = conn.cursor()\n"
            "val = 'alice'\n"
            'cursor.execute("SELECT * FROM t WHERE x = \'%s\'" % val)\n'
        )
        r = detect_vulnerability(code)
        self.assertTrue(
            r["is_vulnerable"],
            msg=(
                "整管线未能捕获 `execute(\"...'%s'\" % val)` 字符串格式化漏洞。"
                f" bandit={r['bandit']}, rule_based={r['rule_based']}"
            ),
        )
        self.assertTrue(
            r["bandit"]["b608_hit"],
            msg=(
                "Bandit B608 未命中用户原样 UNSAFE 示例——"
                "删除 percent_execute_tuple 的前提失效，需要重新审视修复方案。"
            ),
        )


if __name__ == "__main__":
    unittest.main()
