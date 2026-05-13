"""对比 7 组实验结果并计算相对 baseline 的 valid-only SQL 注入率下降百分比。

2026-04-21 invalid-extraction 语义加固后的新契约
--------------------------------------------------

**只展示** 两类严格定义过的指标：

1. ``extraction_failure_rate = n_invalid / n_samples``：模型输出中抽取失败的比例；
2. ``sql_injection_rate_valid``：**仅** valid 样本上的 SQL 注入率，以及对应 P/R/F1/FPR/FNR。

并额外暴露 **conservative_metrics** / **strict_metrics** 的 P/R/F1，以让下游在同一
张表里直接看到"抽取失败到底让模型付出了多少代价"。

**已移除**（旧版存在严重漏洞的字段）：

- ``sql_injection_rate`` / ``safe_code_generation_rate``（= vuln 总数 / 全部样本）；
  这两者把 invalid 样本混成"安全"计入，等价于奖励输出乱码，本脚本不再读它们。
- ``classification_vs_expected``：旧版 valid-only 分类块的名字，被 ``valid_only_metrics``
  取代（同时暴露 conservative / strict 两种严苛口径）。

读取的 *_results.json 来自 ``evaluation/evaluator.py::save_results`` 的新 schema，
不兼容 2026-04-20 之前产出的老 JSON——那些老结果请先归档或删除再重跑评测。

2026-04-22 九次加固 —— 响应质量指标进入对比表
--------------------------------------------------

`evaluation/evaluator.py::save_results` 自八次加固起在 ``summary.response_quality_metrics``
里写入 ``warning_rate`` / ``explanation_rate`` / ``safe_solution_rate`` /
``full_compliance_rate``（再加按 ``expected_vulnerable`` 拆分的 8 项子集 rate），但下游
``compare_results.py`` 之前**完全无视**这块——researcher 看不到模型在响应结构层面是
否塌缩，等于让评测重新退化回"只看代码安全"。本次九次加固把响应质量指标纳入对比表
的 first-class 列：

- ``metrics_block_from_eval_json`` 额外抽取 4 项整体 rate；JSON 里**缺字段**时填 ``None``
  而不是 ``raise``，旧版评测 JSON / smoke test 仍能被对比；
- ``_print_table`` 新增 ``warn%`` / ``expl%`` / ``safe%`` / ``full%`` / ``struct%`` 五列，
  以百分比形式（``0.85 → 85.0%``）展示，``None`` 显示为 ``N/A``；
- 顶层 ``comparison_summary`` 同步落入 ``{method}_warning_rate`` /
  ``{method}_explanation_rate`` / ``{method}_safe_solution_rate`` /
  ``{method}_full_compliance_rate`` / ``{method}_structured_response_score``；
- 派生指标 ``structured_response_score = full_compliance_rate`` 用作模型排序锚点
  （三段齐整 = 训练契约的核心信号），同步进入 per_model 块与顶层；
- 既有的 ``sql_injection_rate_valid`` / ``valid_only`` / ``conservative`` / ``strict`` /
  ``extraction_failure_rate`` 字段名、类型、数值**完全不变**——九次加固严格维持
  "Do NOT remove existing metrics" 这条用户硬约束。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError(
            f"{path} 缺少 summary 字段；请确认 JSON 由新版 evaluator.save_results 写出。"
        )
    summary = data["summary"]
    for required in (
        "n_samples",
        "n_valid",
        "n_invalid",
        "extraction_failure_rate",
        "sql_injection_rate_valid",
        "safe_rate_valid",
        "valid_only_metrics",
        "conservative_metrics",
        "strict_metrics",
    ):
        if required not in summary:
            raise ValueError(
                f"{path} summary 缺少 `{required}`；该结果很可能来自旧版评测器，"
                "请重新运行 evaluation/evaluate.py 生成新 schema 的结果 JSON。"
            )
    return summary


def _bundle_metrics(summary: dict, block_key: str) -> dict:
    """读取 valid_only / conservative / strict 里的 P/R/F1/FPR/FNR/confusion。"""
    block = summary.get(block_key) or {}
    cm = block.get("confusion_matrix") or {}
    return {
        "precision": float(block.get("precision_vulnerable", 0.0)),
        "recall": float(block.get("recall_vulnerable", 0.0)),
        "f1": float(block.get("f1_vulnerable", 0.0)),
        "fpr": float(block.get("false_positive_rate", 0.0)),
        "fnr": float(block.get("false_negative_rate", 0.0)),
        "accuracy_secondary": float(block.get("accuracy_secondary", 0.0)),
        "tp": int(cm.get("TP", 0)),
        "fp": int(cm.get("FP", 0)),
        "tn": int(cm.get("TN", 0)),
        "fn": int(cm.get("FN", 0)),
    }


# ---- 2026-04-22 九次加固：响应质量指标的可选抽取 ----------------------

RESPONSE_QUALITY_RATE_KEYS: tuple[str, ...] = (
    "warning_rate",
    "explanation_rate",
    "safe_solution_rate",
    "full_compliance_rate",
)


def _optional_float(block: dict[str, Any], key: str) -> float | None:
    """从 dict 里软取 float：键不存在或值为 None 都返回 None；类型错也不 raise。

    设计意图：`response_quality_metrics` 是 2026-04-22 八次加固才落入 evaluator JSON 的
    新字段。九次加固对它做 **opt-in 兼容**——旧 JSON / smoke test 不带这块时填 None，
    仍能进入对比表，避免破坏 researcher 已有工作流。"""
    if key not in block:
        return None
    value = block[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _response_quality_block(summary: dict[str, Any]) -> dict[str, float | None]:
    """从 summary 抽 4 项整体响应质量 rate + 派生 ``structured_response_score``。

    - 缺 ``response_quality_metrics`` 整块 → 5 项全 None；
    - 缺单项 rate → 该项 None；
    - ``structured_response_score`` 始终等于 ``full_compliance_rate``（包括 None 透传）。

    说明：返回的 dict 用于 ``per_model[method]["response_quality"]``，再由 ``_print_table``
    与 ``comparison_summary`` 顶层共享，单一源——避免计算/取值漂移。
    """
    rq = summary.get("response_quality_metrics") or {}
    rates = {key: _optional_float(rq, key) for key in RESPONSE_QUALITY_RATE_KEYS}
    rates["structured_response_score"] = rates["full_compliance_rate"]
    return rates


def metrics_block_from_eval_json(path: Path) -> dict:
    """从单模型评测 JSON 读取 invalid-aware 指标块（valid-only + conservative + strict）。

    2026-04-22 九次加固扩展：额外抽取 ``response_quality_metrics`` 的 4 项整体 rate +
    派生 ``structured_response_score``，存入 ``response_quality`` 子块。任一字段缺失即填
    ``None``（不 raise，旧 JSON 兼容）。既有键（``n_samples`` / ``n_valid`` / ``n_invalid`` /
    ``extraction_failure_rate`` / ``sql_injection_rate_valid`` / ``safe_rate_valid`` /
    ``valid_only`` / ``conservative`` / ``strict``）的取值与类型**完全不变**。
    """
    summary = load_summary(path)
    return {
        "n_samples": int(summary["n_samples"]),
        "n_valid": int(summary["n_valid"]),
        "n_invalid": int(summary["n_invalid"]),
        "extraction_failure_rate": float(summary["extraction_failure_rate"]),
        "sql_injection_rate_valid": float(summary["sql_injection_rate_valid"]),
        "safe_rate_valid": float(summary["safe_rate_valid"]),
        # 2026-05-10 修复 F2：主指标进入对比表
        "defense_success_rate": float(summary.get("defense_success_rate", 0.0)),
        "safe_rate_on_benign": float(summary.get("safe_rate_on_benign", 0.0)),
        "valid_only": _bundle_metrics(summary, "valid_only_metrics"),
        "conservative": _bundle_metrics(summary, "conservative_metrics"),
        "strict": _bundle_metrics(summary, "strict_metrics"),
        "response_quality": _response_quality_block(summary),
    }


def pct_drop(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


METHODS: tuple[str, ...] = (
    "baseline",
    "lora_sft",
    "lora_dpo",
    "qlora_sft",
    "qlora_dpo",
)

EXCLUDED_DEGENERATE_METHODS: tuple[str, ...] = ("lora_only", "qlora_only")
EXCLUDED_RESULT_FILENAMES: set[str] = {
    "lora_only_results.json",
    "qlora_only_results.json",
}


def _load_per_sample(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    per_sample = data.get("per_sample")
    if not isinstance(per_sample, list):
        raise ValueError(f"{path} 缺少 per_sample 列表，无法进行退化模型检测。")
    return per_sample


def _sample_signature(sample: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        sample.get("raw_output"),
        sample.get("code"),
        sample.get("is_vulnerable"),
    )


def _exclude_degenerate_duplicates(
    model_blocks: dict[str, dict], model_to_path: dict[str, Path]
) -> tuple[dict[str, dict], list[str]]:
    """比较前做模型去重：输出签名完全一致则视为退化模型并排除。"""
    signatures: dict[str, tuple[tuple[Any, Any, Any], ...]] = {}
    for method, path in model_to_path.items():
        per_sample = _load_per_sample(path)
        signatures[method] = tuple(_sample_signature(s) for s in per_sample)

    dropped: list[str] = []
    methods = list(model_blocks.keys())
    for i, keep in enumerate(methods):
        if keep in dropped:
            continue
        for dup in methods[i + 1 :]:
            if dup in dropped:
                continue
            keep_sig = signatures.get(keep)
            dup_sig = signatures.get(dup)
            if not keep_sig or not dup_sig:
                continue
            if keep_sig == dup_sig:
                warnings.warn(
                    f"Degenerate model detected (identical outputs): {dup} == {keep}"
                )
                dropped.append(dup)

    if not dropped:
        return model_blocks, dropped

    filtered = {m: b for m, b in model_blocks.items() if m not in set(dropped)}
    return filtered, dropped


def _print_header(cols: list[str]) -> None:
    print(" | ".join(cols))
    print("-+-".join("-" * len(c) for c in cols))


def _format_pct_cell(value: float | None, width: int) -> str:
    """把 0~1 的 rate 渲染成右对齐的百分比字符串：0.85 → ' 85.0%'；None → '   N/A'。

    与既有 ``f1_valid`` 等列采用 ``{:.4f}`` 浮点格式不同，响应质量列**故意**走百分比
    形式——researcher 在论文 / slides 里读到 ``87.0%`` 比 ``0.8700`` 直观，且与 README
    《响应质量指标》一节里所有 ``0.X 应高 / 0.Y 应低`` 的契约表达式直接对齐。
    """
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value * 100.0:>{width - 1}.1f}%"


def _print_table(per_model: dict[str, dict]) -> None:
    """打印有效模型对比表（默认 5 个方法，退化重复模型会进一步减少）。

    既有列（**完全不变**）：
        model | n_samples | n_invalid | ext_fail | inj_valid | F1_valid | F1_cons | F1_strict

    2026-04-22 九次加固新增 5 列（`%` 后缀，None 显示为 N/A）：
        warn% | expl% | safe% | full% | struct%

    其中 ``struct% = full% = full_compliance_rate * 100``，作为模型排序锚点（三段
    齐整率），与 README《响应质量指标（2026-04-22 八次加固）》一节里的契约方向严格
    一致：训练契约良好的模型在 ``full% / struct%`` 列上应显著高于 baseline。
    """
    header = [
        f"{'model':<12}",
        f"{'n_samples':>9}",
        f"{'n_invalid':>9}",
        f"{'ext_fail':>8}",
        f"{'defense%':>8}",
        f"{'benign%':>8}",
        f"{'inj_valid':>9}",
        f"{'F1_valid':>9}",
        f"{'F1_cons':>8}",
        f"{'F1_strict':>9}",
        f"{'warn%':>7}",
        f"{'expl%':>7}",
        f"{'safe%':>7}",
        f"{'full%':>7}",
        f"{'struct%':>7}",
    ]
    _print_header(header)
    for method, row in per_model.items():
        rq = row.get("response_quality") or {}
        cols = [
            f"{method:<12}",
            f"{row['n_samples']:>9d}",
            f"{row['n_invalid']:>9d}",
            f"{row['extraction_failure_rate']:>8.4f}",
            _format_pct_cell(row.get("defense_success_rate"), 8),
            _format_pct_cell(row.get("safe_rate_on_benign"), 8),
            f"{row['sql_injection_rate_valid']:>9.4f}",
            f"{row['valid_only']['f1']:>9.4f}",
            f"{row['conservative']['f1']:>8.4f}",
            f"{row['strict']['f1']:>9.4f}",
            _format_pct_cell(rq.get("warning_rate"), 7),
            _format_pct_cell(rq.get("explanation_rate"), 7),
            _format_pct_cell(rq.get("safe_solution_rate"), 7),
            _format_pct_cell(rq.get("full_compliance_rate"), 7),
            _format_pct_cell(rq.get("structured_response_score"), 7),
        ]
        print(" | ".join(cols))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="跳过缺失的结果文件（baseline 必须存在）",
    )
    args = p.parse_args()

    # 2026-05-13: 合并 default.yaml（消融实验兼容）
    from training.config_utils import load_merged_config as _lcm
    cfg = _lcm(ROOT, args.config)

    outs = cfg["outputs"]
    method_to_output = {
        "baseline": outs["baseline_results"],
        "lora_sft": outs["lora_sft_results"],
        "lora_dpo": outs["lora_dpo_results"],
        "qlora_sft": outs["qlora_sft_results"],
        "qlora_dpo": outs["qlora_dpo_results"],
    }
    for excluded_method in EXCLUDED_DEGENERATE_METHODS:
        excluded_rel = outs.get(f"{excluded_method}_results")
        if excluded_rel and Path(excluded_rel).name in EXCLUDED_RESULT_FILENAMES:
            warnings.warn(
                f"Skip degenerate result file: {Path(excluded_rel).name} ({excluded_method})"
            )

    missing = [m for m, p in method_to_output.items() if not (ROOT / p).exists()]
    if "baseline" in missing:
        raise FileNotFoundError("缺少 baseline 结果，无法汇总")
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"缺少实验结果文件（或加 --allow-missing）: {missing}"
        )

    base_summary = load_summary(ROOT / method_to_output["baseline"])
    baseline_inj_valid = float(base_summary["sql_injection_rate_valid"])
    baseline_defense = float(base_summary.get("defense_success_rate", 0.0))
    baseline_benign = float(base_summary.get("safe_rate_on_benign", 0.0))
    baseline_rq = _response_quality_block(base_summary)

    summary: dict = {
        "eval_dataset": cfg["files"]["eval_prompts"],
        "baseline_extraction_failure_rate": float(base_summary["extraction_failure_rate"]),
        "baseline_sql_injection_rate_valid": baseline_inj_valid,
        "baseline_safe_rate_valid": float(base_summary["safe_rate_valid"]),
        # 2026-05-10 修复 F2：主指标进入顶层 summary
        "baseline_defense_success_rate": baseline_defense,
        "baseline_safe_rate_on_benign": baseline_benign,
        # 2026-04-22 九次加固：baseline 的响应质量基线
        "baseline_warning_rate": baseline_rq["warning_rate"],
        "baseline_explanation_rate": baseline_rq["explanation_rate"],
        "baseline_safe_solution_rate": baseline_rq["safe_solution_rate"],
        "baseline_full_compliance_rate": baseline_rq["full_compliance_rate"],
        "baseline_structured_response_score": baseline_rq["structured_response_score"],
    }
    per_model: dict[str, dict] = {}
    loaded_paths: dict[str, Path] = {}
    for method in METHODS:
        rel = method_to_output[method]
        if not (ROOT / rel).exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(ROOT / rel)
        block = metrics_block_from_eval_json(ROOT / rel)
        loaded_paths[method] = ROOT / rel
        summary[f"{method}_extraction_failure_rate"] = block["extraction_failure_rate"]
        summary[f"{method}_sql_injection_rate_valid"] = block["sql_injection_rate_valid"]
        summary[f"{method}_safe_rate_valid"] = block["safe_rate_valid"]
        # 2026-05-10 修复 F2：主指标进入顶层 summary
        summary[f"{method}_defense_success_rate"] = block["defense_success_rate"]
        summary[f"{method}_safe_rate_on_benign"] = block["safe_rate_on_benign"]
        summary[f"{method}_sql_injection_reduction_valid_vs_baseline_pct"] = pct_drop(
            baseline_inj_valid, block["sql_injection_rate_valid"]
        )
        # 2026-04-22 九次加固：4 项响应质量 rate + 派生 structured_response_score
        # 同步落到顶层，方便 jq / pandas / 下游脚本按 "{method}_<rate>" 取值
        rq = block["response_quality"]
        summary[f"{method}_warning_rate"] = rq["warning_rate"]
        summary[f"{method}_explanation_rate"] = rq["explanation_rate"]
        summary[f"{method}_safe_solution_rate"] = rq["safe_solution_rate"]
        summary[f"{method}_full_compliance_rate"] = rq["full_compliance_rate"]
        summary[f"{method}_structured_response_score"] = rq["structured_response_score"]
        per_model[method] = block

    per_model, dropped_methods = _exclude_degenerate_duplicates(per_model, loaded_paths)
    for dropped in dropped_methods:
        summary.pop(f"{dropped}_extraction_failure_rate", None)
        summary.pop(f"{dropped}_sql_injection_rate_valid", None)
        summary.pop(f"{dropped}_safe_rate_valid", None)
        summary.pop(f"{dropped}_defense_success_rate", None)
        summary.pop(f"{dropped}_safe_rate_on_benign", None)
        summary.pop(f"{dropped}_sql_injection_reduction_valid_vs_baseline_pct", None)
        summary.pop(f"{dropped}_warning_rate", None)
        summary.pop(f"{dropped}_explanation_rate", None)
        summary.pop(f"{dropped}_safe_solution_rate", None)
        summary.pop(f"{dropped}_full_compliance_rate", None)
        summary.pop(f"{dropped}_structured_response_score", None)
    summary["per_model"] = per_model

    out_file = ROOT / outs.get("comparison_summary", "outputs/comparison_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("")
    print("=== Summary table (valid-only / conservative / strict + response quality) ===")
    _print_table(per_model)
    print(
        "[Legend] warn% / expl% / safe% / full% = response_quality_metrics 整体 rate；"
        "struct% = structured_response_score (= full_compliance_rate)。"
        "缺字段（旧版 evaluator JSON）显示为 N/A。"
    )
    print(f"[OK] wrote {out_file}")

    compare_alt = outs.get("compare_results")
    if compare_alt:
        alt_path = ROOT / compare_alt
        alt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {alt_path}")


if __name__ == "__main__":
    main()
