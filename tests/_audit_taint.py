"""
临时诊断脚本：验证 taint tracker 是否真的在工作
测试 taint 管线从 detect_vulnerability → run_taint_analysis 的完整链路
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detection.sql_injection_detector import detect_vulnerability
from detection.taint_tracker import run_taint_analysis


def test_taint_standalone() -> None:
    print("=" * 60)
    print("TAINT TRACKER DIAGNOSIS")
    print("=" * 60)

    # Test 1: 已知应该触发 taint 的代码（来自 test_taint_tracker.py）
    code_vuln = """
user = taint_input("admin")
query = f"SELECT * FROM users WHERE name = '{user}'"
import sqlite3
conn = sqlite3.connect(":memory:")
conn.execute(query)
"""
    print("\n[Test 1] standalone run_taint_analysis on vulnerable code:")
    r = run_taint_analysis(code_vuln)
    print(f"  is_vulnerable={r['is_vulnerable']}")
    print(f"  taint_flows_detected={r['taint_flows_detected']}")
    print(f"  error={r.get('error')}")
    print(f"  details={r.get('details')}")
    status1 = "✅" if r["is_vulnerable"] and r["taint_flows_detected"] > 0 else "❌ FAIL"
    print(f"  Status: {status1}")

    # Test 2: 安全参数化查询
    code_safe = """
import sqlite3
conn = sqlite3.connect(":memory:")
conn.execute("SELECT ? AS x", ("ok",))
"""
    print("\n[Test 2] standalone run_taint_analysis on safe code:")
    r2 = run_taint_analysis(code_safe)
    print(f"  is_vulnerable={r2['is_vulnerable']}")
    print(f"  taint_flows_detected={r2['taint_flows_detected']}")
    print(f"  error={r2.get('error')}")
    status2 = "✅" if not r2["is_vulnerable"] else "❌ FALSE POSITIVE"
    print(f"  Status: {status2}")

    # Test 3: 通过 detect_vulnerability（enable_taint=True）
    print("\n[Test 3] detect_vulnerability with enable_taint=True on vulnerable code:")
    d = detect_vulnerability(code_vuln, enable_taint=True)
    taint_block = d.get("taint", {})
    print(f"  taint.is_vulnerable={taint_block.get('is_vulnerable')}")
    print(f"  taint.taint_flows_detected={taint_block.get('taint_flows_detected')}")
    print(f"  taint.skipped={taint_block.get('skipped')}")
    print(f"  overall is_vulnerable={d.get('is_vulnerable')}")
    print(f"  detection_sources={d.get('detection_sources')}")
    status3 = "✅" if "taint" in d.get("detection_sources", []) else "⚠ taint not in sources"
    print(f"  Status: {status3}")

    # Test 4: 通过 detect_vulnerability（enable_taint=False，默认）
    print("\n[Test 4] detect_vulnerability with enable_taint=False (DEFAULT) on vulnerable code:")
    d2 = detect_vulnerability(code_vuln, enable_taint=False)
    taint_block2 = d2.get("taint", {})
    print(f"  taint.skipped={taint_block2.get('skipped')}")
    print(f"  taint.is_vulnerable={taint_block2.get('is_vulnerable')}")
    print(f"  detection_sources={d2.get('detection_sources')}")
    status4 = "✅ (taint correctly skipped)" if taint_block2.get("skipped") else "❌ UNEXPECTED"
    print(f"  Status: {status4}")

    # Test 5: baseline 中常见的安全代码模式（参数化 pymysql）
    code_baseline_style = """
import pymysql

def good(cur, uid):
    cur.execute("SELECT * FROM users WHERE user_id = %s", (uid,))
    return cur.fetchall()
"""
    print("\n[Test 5] baseline-style safe code with taint:")
    r5 = run_taint_analysis(code_baseline_style)
    print(f"  is_vulnerable={r5['is_vulnerable']}")
    print(f"  taint_flows_detected={r5['taint_flows_detected']}")
    print(f"  error={r5.get('error')}")
    status5 = "✅" if not r5["is_vulnerable"] else "❌ FALSE POSITIVE"
    print(f"  Status: {status5}")

    # Test 6: 从 baseline_results.json 中拿一条真实 code 跑 taint
    import json
    baseline = ROOT / "outputs" / "baseline_results.json"
    if baseline.exists():
        data = json.loads(baseline.read_text(encoding="utf-8"))
        valid_samples = [s for s in data["per_sample"] if not s.get("invalid_extraction")]
        # 找一条 bandit 命中但 taint 可能漏的
        for s in valid_samples:
            if s.get("bandit_detected") and not s.get("taint_detected"):
                code = s.get("code", "")
                if len(code) > 10:
                    print(f"\n[Test 6] Real baseline sample id={s['id']} (bandit=YES, taint=NO):")
                    print(f"  code[:200]={code[:200]!r}")
                    r6 = run_taint_analysis(code)
                    print(f"  taint result: is_vuln={r6['is_vulnerable']} flows={r6['taint_flows_detected']} error={r6.get('error')}")
                    # 分析为什么 taint 没检测到
                    if not r6["is_vulnerable"] and not r6.get("error"):
                        print(f"  ⚠ Taint missed this! Bandit found it but taint didn't.")
                        print(f"  Reason: taint requires code to be EXECUTABLE in restricted sandbox.")
                        print(f"  The code may reference external modules (pymysql) that taint sandbox blocks.")
                    break

    print("\n" + "=" * 60)
    print("TAINT DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_taint_standalone()
