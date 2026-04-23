"""污点追踪单元测试。"""
from __future__ import annotations

import unittest

from detection.sql_injection_detector import detect_vulnerability
from detection.taint_tracker import run_taint_analysis, taint_input, TaintedStr


class TestTaintTracker(unittest.TestCase):
    def test_fstring_to_sqlite_execute_detects_taint(self) -> None:
        code = """
user = taint_input("admin")
query = f"SELECT * FROM users WHERE name = '{user}'"
import sqlite3
conn = sqlite3.connect(":memory:")
conn.execute(query)
"""
        r = run_taint_analysis(code)
        self.assertIsNone(r.get("error"), msg=r.get("error"))
        self.assertTrue(r["is_vulnerable"])
        self.assertGreaterEqual(r["taint_flows_detected"], 1)

    def test_safe_parameterized_no_taint_flow(self) -> None:
        code = """
import sqlite3
conn = sqlite3.connect(":memory:")
conn.execute("SELECT ? AS x", ("ok",))
"""
        r = run_taint_analysis(code)
        self.assertFalse(r["is_vulnerable"])
        self.assertEqual(r["taint_flows_detected"], 0)

    def test_tainted_str_concat_propagates(self) -> None:
        a = TaintedStr("x", True)
        b = a + "y"
        self.assertIsInstance(b, TaintedStr)
        self.assertTrue(b.tainted)

    def test_taint_input_marks_tainted(self) -> None:
        u = taint_input("u")
        self.assertTrue(u.tainted)

    def test_detect_vulnerability_merges_taint_when_enabled(self) -> None:
        code = """
user = taint_input("admin")
query = f"SELECT * FROM users WHERE name = '{user}'"
import sqlite3
conn = sqlite3.connect(":memory:")
conn.execute(query)
"""
        r = detect_vulnerability(code, enable_taint=True, enable_rule_based=False)
        self.assertTrue(r["taint"]["is_vulnerable"])
        self.assertTrue(r["is_vulnerable"])


if __name__ == "__main__":
    unittest.main()
