"""从各模型评测 JSON 生成柱状图：valid-only SQL 注入率 / extraction_failure_rate / FPR+FNR。

2026-04-21 invalid-extraction 语义加固版：严格只用 valid 样本的 SQL 注入率
（``sql_injection_rate_valid``）与 ``valid_only_metrics.false_*_rate``，
并额外画出每个模型的 extraction_failure_rate 柱——让"模型靠乱码白嫖分数"的异常
一眼可见。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS: tuple[str, ...] = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
)


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError(
            f"{path} 缺少 summary 字段；请使用新版 evaluator.save_results 生成的结果 JSON。"
        )
    summary = data["summary"]
    for required in (
        "sql_injection_rate_valid",
        "extraction_failure_rate",
        "valid_only_metrics",
    ):
        if required not in summary:
            raise ValueError(
                f"{path} 缺少 `{required}` 字段；该结果文件 schema 过旧，"
                "请重跑 evaluation/evaluate.py 重新产出。"
            )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--output-dir",
        default="outputs/plots",
        help=(
            "输出目录（将写入 sql_injection_rate_valid.png / extraction_failure_rate.png / "
            "fpr_fnr_valid.png）"
        ),
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="缺失的结果文件跳过该柱，不全则仍出图",
    )
    args = p.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    outs = cfg["outputs"]
    method_to_output = {
        "baseline": outs["baseline_results"],
        "lora_only": outs["lora_only_results"],
        "lora_sft": outs["lora_sft_results"],
        "lora_dpo": outs["lora_dpo_results"],
        "qlora_only": outs["qlora_only_results"],
        "qlora_sft": outs["qlora_sft_results"],
        "qlora_dpo": outs["qlora_dpo_results"],
    }

    labels: list[str] = []
    inj_valid: list[float] = []
    ext_fail: list[float] = []
    fprs: list[float] = []
    fnrs: list[float] = []

    for m in METHODS:
        rel = method_to_output[m]
        path = ROOT / rel
        if not path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"missing results: {path}")
        summ = _load_summary(path)
        valid_only = summ.get("valid_only_metrics", {}) or {}
        labels.append(m)
        inj_valid.append(float(summ["sql_injection_rate_valid"]))
        ext_fail.append(float(summ["extraction_failure_rate"]))
        fprs.append(float(valid_only.get("false_positive_rate", 0.0)))
        fnrs.append(float(valid_only.get("false_negative_rate", 0.0)))

    if not labels:
        raise SystemExit("no result files found")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(labels, inj_valid)
    ax1.set_ylabel("rate")
    ax1.set_title("sql_injection_rate_valid (merged detector, valid samples only)")
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()
    fig1.savefig(out_dir / "sql_injection_rate_valid.png")
    plt.close(fig1)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(labels, ext_fail, color="crimson")
    ax3.set_ylabel("rate")
    ax3.set_ylim(0.0, 1.0)
    ax3.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax3.set_title("extraction_failure_rate (>0.5 会让 evaluator RuntimeError)")
    ax3.tick_params(axis="x", rotation=45)
    fig3.tight_layout()
    fig3.savefig(out_dir / "extraction_failure_rate.png")
    plt.close(fig3)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x = range(len(labels))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], fprs, width=w, label="FPR (valid)")
    ax2.bar([i + w / 2 for i in x], fnrs, width=w, label="FNR (valid)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("rate")
    ax2.set_title("valid_only FPR / FNR vs expected_vulnerable")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "fpr_fnr_valid.png")
    plt.close(fig2)

    print(f"[OK] wrote {out_dir / 'sql_injection_rate_valid.png'}")
    print(f"[OK] wrote {out_dir / 'extraction_failure_rate.png'}")
    print(f"[OK] wrote {out_dir / 'fpr_fnr_valid.png'}")


if __name__ == "__main__":
    main()
