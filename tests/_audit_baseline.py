"""
临时审计脚本：深入检查 baseline_results.json 的 per_sample 数据质量
不修改任何工程代码，仅做只读诊断。
"""
from __future__ import annotations

import json
import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> None:
    baseline = ROOT / "outputs" / "baseline_results.json"
    if not baseline.exists():
        print(f"ERROR: {baseline} not found")
        return

    data = json.loads(baseline.read_text(encoding="utf-8"))
    samples = data["per_sample"]
    summary = data["summary"]

    print("=" * 60)
    print("BASELINE RESULTS AUDIT")
    print("=" * 60)

    # 1. 全局统计
    n_total = len(samples)
    n_invalid = sum(1 for s in samples if s.get("invalid_extraction"))
    n_valid = n_total - n_invalid
    print(f"\n[1] Global stats:")
    print(f"    total={n_total} valid={n_valid} invalid={n_invalid}")
    print(f"    extraction_failure_rate={n_invalid/n_total:.4f}")

    # 2. 对 valid 样本的 code 做 ast.parse 二次验证
    print(f"\n[2] Double-checking AST validity of 'valid' samples:")
    bogus_valid = 0
    empty_code = 0
    for s in samples:
        if s.get("invalid_extraction"):
            continue
        code = s.get("code", "")
        if not code or not code.strip():
            empty_code += 1
            print(f"    EMPTY CODE: id={s['id']!r}")
            continue
        try:
            ast.parse(code)
        except SyntaxError as exc:
            bogus_valid += 1
            print(f"    SYNTAX ERROR: id={s['id']!r} line={exc.lineno} msg={exc.msg}")
            print(f"    code[:200]={code[:200]!r}")

    print(f"    empty_code_in_valid={empty_code} bogus_ast_in_valid={bogus_valid}")
    if empty_code + bogus_valid > 0:
        print(f"    ⚠ REAL invalid rate >= {(n_invalid+empty_code+bogus_valid)/n_total:.4f}")

    # 3. 检查 code 内容质量
    print(f"\n[3] Code quality sampling (first 10 valid):")
    count = 0
    for s in samples:
        if s.get("invalid_extraction"):
            continue
        code = s.get("code", "")
        print(f"\n  id={s['id']}")
        print(f"  expected_vulnerable={s['expected_vulnerable']}")
        print(f"  is_vulnerable={s['is_vulnerable']}")
        print(f"  bandit={s['bandit_detected']} rule={s['rule_based_detected']} taint={s['taint_detected']}")
        print(f"  code_len={len(code)}")
        print(f"  code[:300]={code[:300]!r}")
        count += 1
        if count >= 10:
            break

    # 4. 检查 detector 实际参与度
    print(f"\n[4] Detector participation:")
    for s in samples:
        if s.get("invalid_extraction"):
            continue
    valid_samples = [s for s in samples if not s.get("invalid_extraction")]
    print(f"    bandit_detected={sum(1 for s in valid_samples if s['bandit_detected'])}/{len(valid_samples)}")
    print(f"    rule_based_detected={sum(1 for s in valid_samples if s['rule_based_detected'])}/{len(valid_samples)}")
    print(f"    taint_detected={sum(1 for s in valid_samples if s['taint_detected'])}/{len(valid_samples)}")
    print(f"    taint_flows>0={sum(1 for s in valid_samples if s.get('taint_flows_detected', 0) > 0)}/{len(valid_samples)}")

    # 5. 检查输出是否包含训练提示
    print(f"\n[5] Prompt leakage check:")
    leak_count = 0
    for s in valid_samples:
        raw = str(s.get("raw_output", ""))
        if "Instruction:" in raw or "Input:" in raw or "###" in raw:
            leak_count += 1
    print(f"    samples with prompt leakage: {leak_count}/{len(valid_samples)}")

    # 6. 重复输出检测
    print(f"\n[6] Repetition detection:")
    code_counter = Counter(s.get("code", "") for s in valid_samples if s.get("code", "").strip())
    duplicates = {c: n for c, n in code_counter.items() if n > 1}
    print(f"    unique codes: {len(code_counter)}/{len(valid_samples)}")
    print(f"    duplicated codes: {len(duplicates)} patterns, total instances: {sum(duplicates.values())}")

    # 7. 按 expected_vulnerable 分组统计
    print(f"\n[7] By expected_vulnerable:")
    for ev in (True, False):
        subset = [s for s in valid_samples if s["expected_vulnerable"] == ev]
        vuln = sum(1 for s in subset if s["is_vulnerable"])
        print(f"    expected_vulnerable={ev}: n={len(subset)} is_vulnerable={vuln} rate={vuln/len(subset):.4f}" if subset else f"    expected_vulnerable={ev}: empty")

    # 8. 对比 summary 中的 aggregated 数据和 per_sample 真实数据
    print(f"\n[8] Summary vs per_sample consistency:")
    summary_tp = summary["valid_only_metrics"]["confusion_matrix"]["TP"]
    summary_fp = summary["valid_only_metrics"]["confusion_matrix"]["FP"]
    summary_tn = summary["valid_only_metrics"]["confusion_matrix"]["TN"]
    summary_fn = summary["valid_only_metrics"]["confusion_matrix"]["FN"]

    # 从 per_sample 重算
    tp = fp = tn = fn = 0
    for s in valid_samples:
        ev = s["expected_vulnerable"]
        iv = s["is_vulnerable"]
        if ev and iv: tp += 1
        elif ev and not iv: fn += 1
        elif not ev and iv: fp += 1
        else: tn += 1

    print(f"    summary: TP={summary_tp} FP={summary_fp} TN={summary_tn} FN={summary_fn}")
    print(f"    recomputed: TP={tp} FP={fp} TN={tn} FN={fn}")
    if (tp, fp, tn, fn) != (summary_tp, summary_fp, summary_tn, summary_fn):
        print(f"    ⚠ MISMATCH! Data corruption or eval skew detected.")
    else:
        print(f"    ✅ Consistent.")

    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
