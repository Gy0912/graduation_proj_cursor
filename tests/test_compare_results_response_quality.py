"""验收：2026-04-22 九次加固——`scripts/compare_results.py` 把响应质量指标
（`warning_rate` / `explanation_rate` / `safe_solution_rate` / `full_compliance_rate`
+ 派生 `structured_response_score`）作为 first-class 列纳入对比表与 JSON 汇总。

覆盖 5 个维度：

1. **per-model 抽取**：`metrics_block_from_eval_json` 读取 `summary.response_quality_metrics`
   的 4 项整体 rate、并派生 `structured_response_score = full_compliance_rate`；
2. **缺字段兼容**（**必须不 raise**）：`response_quality_metrics` 整块缺失 / 单项缺失 /
   值为 None / 类型错时，对应字段填 `None` 而**不是**抛异常；
3. **不同模型差异**：跨多模型对比时，response_quality 数值应**确实不同**——这是用户
   VALIDATION 的核心要求："response metrics differ across models, values are not all
   identical, high-quality models show higher compliance"；
4. **既有指标不变**：新增字段后，`extraction_failure_rate` / `sql_injection_rate_valid` /
   `valid_only.f1` / `conservative.f1` / `strict.f1` / `n_samples` / `n_valid` /
   `n_invalid` 数值与类型**完全不变**（与八次加固在 `metrics.py` 一侧的 `TestExistingMetricsUnchanged`
   遥相呼应：九次加固只动 `compare_results.py`，既有列必须不漂移）；
5. **顶层 summary 注入**：`{method}_warning_rate` / `{method}_explanation_rate` /
   `{method}_safe_solution_rate` / `{method}_full_compliance_rate` /
   `{method}_structured_response_score` 5 个新键写入 `comparison_summary.json`，
   且既有的 `{method}_sql_injection_rate_valid` / `{method}_extraction_failure_rate` /
   `{method}_safe_rate_valid` / `{method}_sql_injection_reduction_valid_vs_baseline_pct`
   全部保留。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_results as cr


# ---------- per-model JSON fixture：合成一份完整 schema 的评测 JSON ----------


def _make_summary_payload(
    *,
    n_samples: int = 600,
    n_valid: int = 540,
    n_invalid: int = 60,
    extraction_failure_rate: float = 0.10,
    sql_injection_rate_valid: float = 0.28,
    safe_rate_valid: float = 0.72,
    f1_valid: float = 0.62,
    f1_cons: float = 0.58,
    f1_strict: float = 0.55,
    response_quality: dict | None = None,
) -> dict:
    """构造满足 `load_summary` 所有 9 个必填字段的最小 summary。

    `response_quality` 如果为 None，则**完全不写** `response_quality_metrics` 块——
    模拟 2026-04-22 八次加固之前的旧 evaluator JSON。
    """
    summary: dict = {
        "n_samples": n_samples,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "extraction_failure_rate": extraction_failure_rate,
        "sql_injection_rate_valid": sql_injection_rate_valid,
        "safe_rate_valid": safe_rate_valid,
        "valid_only_metrics": {
            "confusion_matrix": {"TP": 153, "FP": 100, "TN": 200, "FN": 87},
            "precision_vulnerable": 0.6047,
            "recall_vulnerable": 0.6375,
            "f1_vulnerable": f1_valid,
            "false_positive_rate": 0.3333,
            "false_negative_rate": 0.3625,
            "accuracy_secondary": 0.6537,
        },
        "conservative_metrics": {
            "confusion_matrix": {"TP": 153, "FP": 100, "TN": 227, "FN": 120},
            "precision_vulnerable": 0.6047,
            "recall_vulnerable": 0.5604,
            "f1_vulnerable": f1_cons,
            "false_positive_rate": 0.3058,
            "false_negative_rate": 0.4396,
            "accuracy_secondary": 0.6333,
        },
        "strict_metrics": {
            "confusion_matrix": {"TP": 153, "FP": 127, "TN": 200, "FN": 120},
            "precision_vulnerable": 0.5464,
            "recall_vulnerable": 0.5604,
            "f1_vulnerable": f1_strict,
            "false_positive_rate": 0.3884,
            "false_negative_rate": 0.4396,
            "accuracy_secondary": 0.5883,
        },
    }
    if response_quality is not None:
        summary["response_quality_metrics"] = response_quality
    return {"meta": {"mode": "test"}, "summary": summary, "per_sample": []}


def _full_response_quality(
    *,
    warning_rate: float = 0.45,
    explanation_rate: float = 0.43,
    safe_solution_rate: float = 0.41,
    full_compliance_rate: float = 0.40,
    on_pos: float = 0.85,
    on_neg: float = 0.05,
) -> dict:
    return {
        "n_samples_used": 600,
        "n_positives": 300,
        "n_negatives": 300,
        "warning_rate": warning_rate,
        "explanation_rate": explanation_rate,
        "safe_solution_rate": safe_solution_rate,
        "full_compliance_rate": full_compliance_rate,
        "warning_rate_on_positives": on_pos,
        "explanation_rate_on_positives": on_pos,
        "safe_solution_rate_on_positives": on_pos,
        "full_compliance_rate_on_positives": on_pos,
        "warning_rate_on_negatives": on_neg,
        "explanation_rate_on_negatives": on_neg,
        "safe_solution_rate_on_negatives": on_neg,
        "full_compliance_rate_on_negatives": on_neg,
        "markers": {
            "warning": "[SECURITY WARNING]",
            "explanation": "[EXPLANATION]",
            "safe_solution": "[SAFE SOLUTION]",
        },
        "note": "test fixture",
    }


# ---------- Tests ----------


class TestMetricsBlockExtractsResponseQuality(unittest.TestCase):
    """`metrics_block_from_eval_json` 必须把 4 项整体 rate + struct_score 抽进 ``response_quality`` 子块。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def _write(self, name: str, payload: dict) -> Path:
        p = self.tmp_path / name
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_full_response_quality_block_extracted(self) -> None:
        payload = _make_summary_payload(
            response_quality=_full_response_quality(
                warning_rate=0.91,
                explanation_rate=0.90,
                safe_solution_rate=0.89,
                full_compliance_rate=0.87,
            )
        )
        block = cr.metrics_block_from_eval_json(self._write("a.json", payload))
        rq = block["response_quality"]
        self.assertAlmostEqual(rq["warning_rate"], 0.91)
        self.assertAlmostEqual(rq["explanation_rate"], 0.90)
        self.assertAlmostEqual(rq["safe_solution_rate"], 0.89)
        self.assertAlmostEqual(rq["full_compliance_rate"], 0.87)
        # 派生指标必须等于 full_compliance_rate
        self.assertAlmostEqual(
            rq["structured_response_score"], rq["full_compliance_rate"]
        )

    def test_missing_response_quality_metrics_block_yields_all_none(self) -> None:
        """旧版 evaluator JSON（八次加固前）不带 `response_quality_metrics` —— 必须不 raise。"""
        payload = _make_summary_payload(response_quality=None)
        block = cr.metrics_block_from_eval_json(self._write("b.json", payload))
        rq = block["response_quality"]
        for key in (
            "warning_rate",
            "explanation_rate",
            "safe_solution_rate",
            "full_compliance_rate",
            "structured_response_score",
        ):
            self.assertIsNone(rq[key], f"{key} should be None when block missing")

    def test_partial_missing_keys_yield_per_field_none(self) -> None:
        """单项缺失 → 该项 None，其他项保留实际数值。"""
        partial = _full_response_quality()
        del partial["explanation_rate"]
        del partial["safe_solution_rate"]
        payload = _make_summary_payload(response_quality=partial)
        block = cr.metrics_block_from_eval_json(self._write("c.json", payload))
        rq = block["response_quality"]
        self.assertIsNotNone(rq["warning_rate"])
        self.assertIsNone(rq["explanation_rate"])
        self.assertIsNone(rq["safe_solution_rate"])
        self.assertIsNotNone(rq["full_compliance_rate"])
        self.assertEqual(rq["structured_response_score"], rq["full_compliance_rate"])

    def test_none_value_treated_as_missing(self) -> None:
        """显式 ``"warning_rate": null`` 与缺键等价：填 None，不 raise。"""
        rq_in = _full_response_quality()
        rq_in["warning_rate"] = None
        rq_in["full_compliance_rate"] = None
        payload = _make_summary_payload(response_quality=rq_in)
        block = cr.metrics_block_from_eval_json(self._write("d.json", payload))
        rq = block["response_quality"]
        self.assertIsNone(rq["warning_rate"])
        self.assertIsNone(rq["full_compliance_rate"])
        self.assertIsNone(rq["structured_response_score"])

    def test_non_numeric_value_yields_none_no_crash(self) -> None:
        """非数值（"yes" / 任意字符串）必须被吃掉为 None，不 ValueError、不 TypeError。"""
        rq_in = _full_response_quality()
        rq_in["warning_rate"] = "yes"
        rq_in["explanation_rate"] = []
        payload = _make_summary_payload(response_quality=rq_in)
        block = cr.metrics_block_from_eval_json(self._write("e.json", payload))
        rq = block["response_quality"]
        self.assertIsNone(rq["warning_rate"])
        self.assertIsNone(rq["explanation_rate"])

    def test_existing_block_keys_unchanged(self) -> None:
        """新增字段后，既有 9 个键的取值与类型完全不变（"Do NOT remove existing metrics"）。"""
        payload = _make_summary_payload(response_quality=_full_response_quality())
        block = cr.metrics_block_from_eval_json(self._write("f.json", payload))
        self.assertEqual(block["n_samples"], 600)
        self.assertEqual(block["n_valid"], 540)
        self.assertEqual(block["n_invalid"], 60)
        self.assertAlmostEqual(block["extraction_failure_rate"], 0.10)
        self.assertAlmostEqual(block["sql_injection_rate_valid"], 0.28)
        self.assertAlmostEqual(block["safe_rate_valid"], 0.72)
        self.assertAlmostEqual(block["valid_only"]["f1"], 0.62)
        self.assertAlmostEqual(block["conservative"]["f1"], 0.58)
        self.assertAlmostEqual(block["strict"]["f1"], 0.55)


class TestPrintTableFormatsResponseQuality(unittest.TestCase):
    """`_print_table` 必须把 5 项响应质量指标渲染为百分比（``0.85 → 85.0%``）；None → ``N/A``。"""

    def _capture_table(self, per_model: dict) -> str:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cr._print_table(per_model)
        return buf.getvalue()

    def _row_block(self, **rq_kwargs) -> dict:
        return {
            "n_samples": 600,
            "n_valid": 540,
            "n_invalid": 60,
            "extraction_failure_rate": 0.10,
            "sql_injection_rate_valid": 0.28,
            "safe_rate_valid": 0.72,
            "valid_only": {"f1": 0.62},
            "conservative": {"f1": 0.58},
            "strict": {"f1": 0.55},
            "response_quality": rq_kwargs,
        }

    def test_table_header_includes_all_five_response_quality_columns(self) -> None:
        per_model = {
            "baseline": self._row_block(
                warning_rate=0.10,
                explanation_rate=0.10,
                safe_solution_rate=0.10,
                full_compliance_rate=0.10,
                structured_response_score=0.10,
            )
        }
        out = self._capture_table(per_model)
        # 5 列新 header 全部存在
        self.assertIn("warn%", out)
        self.assertIn("expl%", out)
        self.assertIn("safe%", out)
        self.assertIn("full%", out)
        self.assertIn("struct%", out)
        # 既有列依然在
        for legacy in ("model", "n_samples", "ext_fail", "inj_valid", "F1_valid", "F1_cons", "F1_strict"):
            self.assertIn(legacy, out)

    def test_percentage_formatting_85_pct_one_decimal(self) -> None:
        per_model = {
            "lora_sft": self._row_block(
                warning_rate=0.85,
                explanation_rate=0.84,
                safe_solution_rate=0.83,
                full_compliance_rate=0.80,
                structured_response_score=0.80,
            )
        }
        out = self._capture_table(per_model)
        # 用户验收原话："values shown as percentages (e.g., 0.85 → 85.0%)"
        self.assertIn("85.0%", out)
        self.assertIn("84.0%", out)
        self.assertIn("83.0%", out)
        self.assertIn("80.0%", out)

    def test_none_renders_as_na_no_crash(self) -> None:
        per_model = {
            "legacy": self._row_block(
                warning_rate=None,
                explanation_rate=None,
                safe_solution_rate=None,
                full_compliance_rate=None,
                structured_response_score=None,
            )
        }
        out = self._capture_table(per_model)
        self.assertIn("N/A", out)
        # 既有列依然出数（不被 None 影响）
        self.assertIn("0.6200", out)  # F1_valid
        self.assertIn("0.5500", out)  # F1_strict


class TestModelsDifferAndRanking(unittest.TestCase):
    """跨多模型场景：response metrics 必须**因模型而异**，且高质量模型在 struct% 列上更高。

    对应用户 VALIDATION 节："response metrics differ across models, values are not all
    identical, high-quality models show higher compliance"。"""

    def test_three_models_have_distinct_full_compliance(self) -> None:
        models = {
            "baseline": _full_response_quality(full_compliance_rate=0.10),
            "lora_only": _full_response_quality(full_compliance_rate=0.30),
            "lora_sft": _full_response_quality(full_compliance_rate=0.85),
        }
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            paths = {
                m: tdp / f"{m}.json"
                for m in models
            }
            for m, rq in models.items():
                paths[m].write_text(
                    json.dumps(_make_summary_payload(response_quality=rq)),
                    encoding="utf-8",
                )
            blocks = {m: cr.metrics_block_from_eval_json(paths[m]) for m in models}

        full_comps = [blocks[m]["response_quality"]["full_compliance_rate"] for m in models]
        # 必须有显著差异：std > 0
        self.assertGreater(max(full_comps) - min(full_comps), 0.10)
        # 不能全部相同
        self.assertNotEqual(full_comps[0], full_comps[1])
        self.assertNotEqual(full_comps[1], full_comps[2])

    def test_high_quality_model_ranks_higher_by_structured_response_score(self) -> None:
        """structured_response_score 用作排序锚点：lora_sft (0.85) > lora_only (0.30) > baseline (0.10)。"""
        models = {
            "baseline": _full_response_quality(full_compliance_rate=0.10),
            "lora_only": _full_response_quality(full_compliance_rate=0.30),
            "lora_sft": _full_response_quality(full_compliance_rate=0.85),
        }
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            scores = {}
            for m, rq in models.items():
                p = tdp / f"{m}.json"
                p.write_text(
                    json.dumps(_make_summary_payload(response_quality=rq)),
                    encoding="utf-8",
                )
                blk = cr.metrics_block_from_eval_json(p)
                scores[m] = blk["response_quality"]["structured_response_score"]

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        self.assertEqual([m for m, _ in ranked], ["lora_sft", "lora_only", "baseline"])


class TestSummaryJsonContainsResponseQualityKeys(unittest.TestCase):
    """端到端：`comparison_summary.json` 顶层必须有 5 个新 ``{method}_*`` 键 + 既有键全保留。"""

    def _run_main_and_load_summary(self, methods_with_rq: dict[str, dict]) -> dict:
        """构造一个临时 cfg + 临时 outputs/* 目录，跑一次 `compare_results.main`，回读 summary JSON。

        `methods_with_rq` 的 key 为 METHODS 子集（baseline 必须存在）；value 为该 method
        的 `response_quality_metrics` 块（None 表示该 evaluator JSON 不带响应质量块）。
        未在 `methods_with_rq` 里出现的 METHODS 会**不**写 *_results.json，靠
        `--allow-missing` 跳过；这是为了把测试场景聚焦在指定的 2~3 个模型上。
        """
        all_methods = (
            "baseline",
            "lora_only",
            "lora_sft",
            "lora_dpo",
            "qlora_only",
            "qlora_sft",
            "qlora_dpo",
        )
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            outputs_dir = tdp / "outputs"
            outputs_dir.mkdir()
            # 写指定 method 的 *_results.json；未指定的 method 不写文件（main 会 --allow-missing 跳过）
            for method, rq in methods_with_rq.items():
                rel = f"outputs/{method}_results.json"
                (tdp / rel).write_text(
                    json.dumps(_make_summary_payload(response_quality=rq)),
                    encoding="utf-8",
                )

            # cfg.outputs 必须包含 7 个 method 的 *_results 键（main 用 outs["..._results"] 取路径），
            # 缺文件靠 --allow-missing 跳过；缺 cfg key 会 KeyError
            cfg = {
                "files": {"eval_prompts": "data/combined/eval_fixed.json"},
                "outputs": {
                    f"{m}_results": f"outputs/{m}_results.json" for m in all_methods
                },
            }
            cfg["outputs"]["comparison_summary"] = "outputs/comparison_summary.json"

            cfg_path = tdp / "cfg.yaml"
            import yaml

            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            # 关键：临时把 cr.ROOT 指到 tdp，让相对路径都基于临时根
            original_root = cr.ROOT
            try:
                cr.ROOT = tdp
                # main() 会用 sys.argv 解析 --config / --allow-missing
                original_argv = sys.argv
                try:
                    sys.argv = [
                        "compare_results.py",
                        "--config",
                        str(cfg_path.relative_to(tdp)),
                        "--allow-missing",
                    ]
                    import io
                    import contextlib

                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cr.main()
                finally:
                    sys.argv = original_argv
            finally:
                cr.ROOT = original_root

            return json.loads(
                (tdp / "outputs" / "comparison_summary.json").read_text(encoding="utf-8")
            )

    def test_top_level_summary_has_per_method_response_quality_keys(self) -> None:
        summary = self._run_main_and_load_summary({
            "baseline": _full_response_quality(
                warning_rate=0.10, explanation_rate=0.10,
                safe_solution_rate=0.10, full_compliance_rate=0.10,
            ),
            "lora_sft": _full_response_quality(
                warning_rate=0.90, explanation_rate=0.89,
                safe_solution_rate=0.88, full_compliance_rate=0.85,
            ),
        })
        # 5 个新顶层键 × 2 个 method → 10 个新键全部存在
        for method in ("baseline", "lora_sft"):
            for key in (
                f"{method}_warning_rate",
                f"{method}_explanation_rate",
                f"{method}_safe_solution_rate",
                f"{method}_full_compliance_rate",
                f"{method}_structured_response_score",
            ):
                self.assertIn(key, summary, f"missing top-level key: {key}")

        # 数值差异（核心：模型间不同）
        self.assertNotAlmostEqual(
            summary["baseline_full_compliance_rate"],
            summary["lora_sft_full_compliance_rate"],
        )
        # struct = full
        self.assertEqual(
            summary["lora_sft_structured_response_score"],
            summary["lora_sft_full_compliance_rate"],
        )

    def test_top_level_summary_keeps_existing_metrics(self) -> None:
        """既有 5 个顶层 baseline_* / 4 个 {method}_* 键必须保留（"Do NOT remove existing metrics"）。"""
        summary = self._run_main_and_load_summary({
            "baseline": _full_response_quality(),
            "lora_sft": _full_response_quality(),
        })
        # baseline_* 五个既有键
        for key in (
            "baseline_extraction_failure_rate",
            "baseline_sql_injection_rate_valid",
            "baseline_safe_rate_valid",
        ):
            self.assertIn(key, summary)
        # {method}_* 四个既有键 / method
        for method in ("baseline", "lora_sft"):
            for key in (
                f"{method}_extraction_failure_rate",
                f"{method}_sql_injection_rate_valid",
                f"{method}_safe_rate_valid",
                f"{method}_sql_injection_reduction_valid_vs_baseline_pct",
            ):
                self.assertIn(key, summary)

    def test_legacy_jsons_without_response_quality_dont_crash(self) -> None:
        """`response_quality_metrics` 缺失的旧 JSON 应被吸收为 None，主流程不 crash。"""
        summary = self._run_main_and_load_summary({
            "baseline": None,
            "lora_sft": _full_response_quality(full_compliance_rate=0.80),
        })
        self.assertIsNone(summary["baseline_full_compliance_rate"])
        self.assertIsNone(summary["baseline_structured_response_score"])
        self.assertAlmostEqual(summary["lora_sft_full_compliance_rate"], 0.80)


if __name__ == "__main__":
    unittest.main()
