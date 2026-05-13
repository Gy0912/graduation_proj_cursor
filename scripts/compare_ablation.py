"""
消融实验对比分析与绘图（2026-05-11）。

读取 outputs/ablation/_all_results.json，生成对比表和图表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_JSON = ROOT / "outputs" / "ablation" / "_all_results.json"
PLOTS_DIR = ROOT / "outputs" / "ablation" / "plots"


def load_results() -> list[dict]:
    if not RESULTS_JSON.exists():
        print(f"Results not found: {RESULTS_JSON}")
        return []
    return json.loads(RESULTS_JSON.read_text("utf-8"))


def plot_injection_rate_comparison(results: list[dict]) -> str:
    """图 1: 消融实验 SFT 注入率对比 bar chart。"""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    groups = []
    values = []
    for r in results:
        m = r.get("sft_metrics")
        if m and m.get("sql_injection_rate_valid") is not None:
            groups.append(f"{r['group']}: {r['name']}")
            values.append(m["sql_injection_rate_valid"])

    if not groups:
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(groups)))
    bars = ax.bar(range(len(groups)), values, color=colors)

    # 基线参考线
    baseline_val = next(
        (r["sft_metrics"]["sql_injection_rate_valid"] for r in results
         if r["group"] == "A" and r.get("sft_metrics")), None
    )
    if baseline_val:
        ax.axhline(y=baseline_val, color="blue", linestyle="--", alpha=0.5,
                   label=f"Full Pipeline (A) = {baseline_val:.4f}")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("SQL Injection Rate (valid)")
    ax.set_title("Ablation Study: SFT sql_injection_rate_valid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = str(PLOTS_DIR / "ablation_injection_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_defense_comparison(results: list[dict]) -> str:
    """图 2: 消融实验 defense_success_rate 对比。"""
    groups = []
    sft_vals = []
    dpo_vals = []
    for r in results:
        sm = r.get("sft_metrics")
        dm = r.get("dpo_metrics")
        if sm:
            groups.append(f"{r['group']}")
            sft_vals.append(sm.get("defense_success_rate", 0))
            dpo_vals.append(dm.get("defense_success_rate", 0) if dm else 0)

    if not groups:
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w/2, sft_vals, w, label="SFT", color="steelblue")
    ax.bar(x + w/2, dpo_vals, w, label="DPO", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("defense_success_rate")
    ax.set_title("Ablation Study: defense_success_rate (SFT vs DPO)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = str(PLOTS_DIR / "ablation_defense_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(results: list[dict]) -> str:
    """生成 Markdown 对比报告。"""
    lines = [
        "# 消融实验对比报告",
        "",
        "## SFT 模型指标对比",
        "",
        "| Group | Name | inj_valid | defense | benign | ext_fail |",
        "|-------|------|-----------|---------|--------|----------|",
    ]
    for r in results:
        m = r.get("sft_metrics")
        if m:
            inj = m.get('sql_injection_rate_valid') or 0
            defense = m.get('defense_success_rate') or 0
            benign = m.get('safe_rate_on_benign') or 0
            ext = m.get('extraction_failure_rate') or 0
            lines.append(
                f"| {r['group']} | {r['name']} | {inj:.4f} | "
                f"{defense:.4f} | {benign:.4f} | {ext:.4f} |"
            )

    lines.append("\n## DPO 模型指标对比\n")
    lines.append("| Group | Name | inj_valid | defense | benign | ext_fail |")
    lines.append("|-------|------|-----------|---------|--------|----------|")
    for r in results:
        m = r.get("dpo_metrics")
        if m:
            inj = m.get('sql_injection_rate_valid') or 0
            defense = m.get('defense_success_rate') or 0
            benign = m.get('safe_rate_on_benign') or 0
            ext = m.get('extraction_failure_rate') or 0
            lines.append(
                f"| {r['group']} | {r['name']} (DPO) | {inj:.4f} | "
                f"{defense:.4f} | {benign:.4f} | {ext:.4f} |"
            )

    return "\n".join(lines)


def main() -> None:
    results = load_results()
    if not results:
        print("No ablation results found. Run scripts/run_ablation.py first.")
        return

    # 生成报告
    report = generate_report(results)
    report_path = ROOT / "outputs" / "ablation" / "comparison_report.md"
    report_path.write_text(report, "utf-8")
    print(f"Report: {report_path}")

    # 生成图表
    p1 = plot_injection_rate_comparison(results)
    p2 = plot_defense_comparison(results)
    if p1: print(f"Chart 1: {p1}")
    if p2: print(f"Chart 2: {p2}")


if __name__ == "__main__":
    main()
