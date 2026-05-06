"""
临时审计脚本：深入检查 lora_sft_results.json 的 per_sample 数据质量
不修改任何工程代码，仅做只读诊断。
"""
from __future__ import annotations

import json
import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def audit_results(name: str, path: Path) -> dict:
    print("=" * 60)
    print(f"{name} RESULTS AUDIT")
    print("=" * 60)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return {"exists": False}

    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["per_sample"]
    summary = data.get("summary", {})
    meta = data.get("meta", {})

    n_total = len(samples)
    n_invalid = sum(1 for s in samples if s.get("invalid_extraction"))
    n_valid = n_total - n_invalid
    print(f"\n[Global] total={n_total} valid={n_valid} invalid={n_invalid} rate={n_invalid/n_total:.4f}" if n_total else "\n[Global] empty")
    print(f"[Meta] max_samples={meta.get('max_samples','N/A')} enable_taint={meta.get('enable_taint','N/A')}")

    # 检查 code 有效性
    bogus_valid = 0
    empty_code = 0
    for s in samples:
        if s.get("invalid_extraction"):
            continue
        code = s.get("code", "")
        if not code or not code.strip():
            empty_code += 1
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            bogus_valid += 1
    print(f"[AST] empty_code_in_valid={empty_code} bogus_ast_in_valid={bogus_valid}")

    # Detector 参与度
    valid_samples = [s for s in samples if not s.get("invalid_extraction")]
    if valid_samples:
        print(f"[Detector] bandit={sum(1 for s in valid_samples if s['bandit_detected'])}/{len(valid_samples)}")
        print(f"[Detector] rule={sum(1 for s in valid_samples if s['rule_based_detected'])}/{len(valid_samples)}")
        print(f"[Detector] taint={sum(1 for s in valid_samples if s['taint_detected'])}/{len(valid_samples)}")
        print(f"[Detector] taint_flows>0={sum(1 for s in valid_samples if s.get('taint_flows_detected',0)>0)}/{len(valid_samples)}")

    # 提示泄漏
    leak = sum(1 for s in valid_samples if "Instruction:" in str(s.get("raw_output", "")) or "Input:" in str(s.get("raw_output", "")))
    print(f"[Leak] samples with prompt leakage: {leak}/{len(valid_samples)}")

    # 代码样本展示
    print(f"\n[Code Samples] (first 6 valid):")
    for s in valid_samples[:6]:
        code = s.get("code", "")
        raw = str(s.get("raw_output", ""))
        is_vuln = s["is_vulnerable"]
        expected = s["expected_vulnerable"]
        print(f"  id={s['id']} exp_vuln={expected} is_vuln={is_vuln} code_len={len(code)}")
        print(f"  code[:200]={code[:200]!r}")
        # Check for repetition patterns in raw output
        lines = raw.split("\n")
        unique_lines = len(set(lines))
        if len(lines) > 10 and unique_lines < len(lines) * 0.3:
            print(f"  ⚠ REPETITION: {len(lines)} lines but only {unique_lines} unique ({unique_lines/len(lines):.1%})")
        print()

    # 响应质量
    rq = summary.get("response_quality_metrics", {})
    if rq:
        print(f"[Response Quality] warn={rq.get('warning_rate',0):.4f} expl={rq.get('explanation_rate',0):.4f} safe={rq.get('safe_solution_rate',0):.4f} full={rq.get('full_compliance_rate',0):.4f}")
        print(f"[Response Quality] mode={rq.get('training_mode','N/A')}")

    # 防御成功率
    dsr = summary.get("defense_success_rate", None)
    if dsr is not None:
        print(f"[Defense] defense_success_rate={dsr:.4f}")

    print()
    return {
        "exists": True,
        "n_total": n_total,
        "n_invalid": n_invalid,
        "extraction_failure_rate": n_invalid / n_total if n_total else 0,
        "empty_code": empty_code,
        "bogus_ast": bogus_valid,
        "leak_count": leak,
    }


def main() -> None:
    results = {}

    # 检查所有存在的 results json
    outputs_dir = ROOT / "outputs"
    files_to_check = [
        "baseline_results.json",
        "lora_sft_results.json",
        "lora_only_results.json",
        "qlora_sft_results.json",
        "qlora_only_results.json",
    ]
    for fname in files_to_check:
        fp = outputs_dir / fname
        name = fname.replace("_results.json", "")
        results[name] = audit_results(name, fp)

    # 总结
    print("=" * 60)
    print("OVERALL DIAGNOSIS SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        if r.get("exists"):
            real_invalid = r["n_invalid"] + r["empty_code"] + r["bogus_ast"]
            real_rate = real_invalid / r["n_total"] if r["n_total"] else 0
            flag = "⚠ CRITICAL" if real_rate > 0.3 else ("⚠ WARN" if real_rate > 0.1 else "✅ OK")
            print(f"  {name}: total={r['n_total']} declared_invalid={r['n_invalid']} real_invalid={real_invalid} real_rate={real_rate:.4f} leaks={r['leak_count']} {flag}")
        else:
            print(f"  {name}: MISSING")


if __name__ == "__main__":
    main()
