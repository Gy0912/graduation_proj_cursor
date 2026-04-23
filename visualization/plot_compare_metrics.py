"""从汇总对比 JSON（新版 compare_results schema）绘制各模型的评测柱状图。

2026-04-21 invalid-extraction 语义加固：
- 只读 ``per_model[method]["sql_injection_rate_valid"]`` / ``valid_only`` / ``extraction_failure_rate``；
- 旧扁平字段（``{method}_sql_injection_rate`` / ``{method}_safe_code_generation_rate``）
  已被破坏性移除，不再提供兼容层——请先重跑 ``scripts/compare_results.py`` 生成新 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHOD_ORDER: tuple[str, ...] = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
)


def load_per_model(data: dict) -> dict[str, dict]:
    pm = data.get("per_model")
    if not isinstance(pm, dict) or not pm:
        raise ValueError(
            "compare_results JSON 缺少 per_model 字段；请确认是由新版 "
            "scripts/compare_results.py 产出（2026-04-21 语义加固后 schema）。"
        )
    cleaned: dict[str, dict] = {}
    for method, block in pm.items():
        if not isinstance(block, dict):
            continue
        for required in ("sql_injection_rate_valid", "extraction_failure_rate", "valid_only"):
            if required not in block:
                raise ValueError(
                    f"per_model[{method!r}] 缺少 `{required}`；请重跑 compare_results。"
                )
        cleaned[method] = block
    return cleaned


def _bar_chart(labels: list[str], values: list[float], ylabel: str, out_path: Path,
               hline: float | None = None, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    if title is not None:
        ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="汇总对比 JSON（含 per_model）")
    p.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="图像输出目录（默认 outputs/plots）",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_model = load_per_model(data)
    if not per_model:
        raise ValueError("未找到任何模型指标（per_model 为空）")

    out_dir = Path(args.output_dir)
    labels: list[str] = []
    inj_valid: list[float] = []
    fpr_valid: list[float] = []
    fnr_valid: list[float] = []
    safe_valid: list[float] = []
    ext_fail: list[float] = []
    f1_cons: list[float] = []
    f1_strict: list[float] = []

    for key in METHOD_ORDER:
        if key not in per_model:
            continue
        row = per_model[key]
        vo = row.get("valid_only", {}) or {}
        cons = row.get("conservative", {}) or {}
        strt = row.get("strict", {}) or {}
        labels.append(key)
        inj_valid.append(float(row["sql_injection_rate_valid"]))
        fpr_valid.append(float(vo.get("fpr", 0.0)))
        fnr_valid.append(float(vo.get("fnr", 0.0)))
        safe_valid.append(float(row.get("safe_rate_valid", 1.0 - inj_valid[-1])))
        ext_fail.append(float(row["extraction_failure_rate"]))
        f1_cons.append(float(cons.get("f1", 0.0)))
        f1_strict.append(float(strt.get("f1", 0.0)))

    _bar_chart(
        labels,
        inj_valid,
        "sql_injection_rate_valid",
        out_dir / "injection_rate_valid.png",
        title="SQL injection rate (valid samples only)",
    )
    _bar_chart(
        labels,
        ext_fail,
        "extraction_failure_rate",
        out_dir / "extraction_failure_rate.png",
        hline=0.5,
        title="extraction_failure_rate (>0.5 会触发 evaluator RuntimeError)",
    )
    _bar_chart(
        labels,
        fpr_valid,
        "fpr (valid)",
        out_dir / "fpr_valid.png",
        title="False Positive Rate (valid samples only)",
    )
    _bar_chart(
        labels,
        fnr_valid,
        "fnr (valid)",
        out_dir / "fnr_valid.png",
        title="False Negative Rate (valid samples only)",
    )
    _bar_chart(
        labels,
        safe_valid,
        "safe_rate_valid",
        out_dir / "safe_rate_valid.png",
        title="Safe code generation rate (valid samples only)",
    )
    _bar_chart(
        labels,
        f1_cons,
        "F1 (conservative)",
        out_dir / "f1_conservative.png",
        title="F1 (conservative: invalid → FN if expected else TN)",
    )
    _bar_chart(
        labels,
        f1_strict,
        "F1 (strict)",
        out_dir / "f1_strict.png",
        title="F1 (strict: invalid → FN if expected else FP)",
    )
    print(
        f"[OK] wrote {out_dir / 'injection_rate_valid.png'}, "
        f"{out_dir / 'extraction_failure_rate.png'}, "
        f"{out_dir / 'fpr_valid.png'}, {out_dir / 'fnr_valid.png'}, "
        f"{out_dir / 'safe_rate_valid.png'}, {out_dir / 'f1_conservative.png'}, "
        f"{out_dir / 'f1_strict.png'}"
    )


if __name__ == "__main__":
    main()
