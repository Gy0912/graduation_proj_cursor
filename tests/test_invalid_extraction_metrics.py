"""验收：invalid-extraction 语义加固后的三组指标语义与硬失败阈值。

对应 2026-04-21 修复（`logs/changelog_2026-04-21_invalid_extraction_semantics.md`）。
这些断言覆盖：

- `valid_only_metrics`：仅 valid 样本的混淆矩阵与 P/R/F1；
- `conservative_metrics`：invalid→FN/TN 分流；
- `strict_metrics`：invalid→FN/FP 分流；
- `extraction_failure_rate > 0.5` 必须 RuntimeError（hard failure）；
- invalid 样本把 `is_vulnerable` 写成 bool（False/True）必须 ValueError，堵死旧 bug 复现路径。
"""
from __future__ import annotations

import unittest

from evaluation.metrics import aggregate_metrics, EXTRACTION_FAILURE_HARD_THRESHOLD


BASE_FIELDS = dict(
    bandit_detected=False,
    bandit_b608=False,
    bandit_has_B608=False,
    rule_based_detected=False,
    taint_detected=False,
    bandit_confidence_levels=[],
    # 2026-04-22 八次加固（响应质量指标）：has_warning/has_explanation/has_safe_solution
    # 是 metrics 层的新强制字段。本测试文件聚焦 invalid-extraction 语义，不关心三段式
    # 合规率，因此统一置 False——只要字段存在且类型是 bool，`_compute_response_quality_*`
    # 的契约即可满足，不影响既有 valid_only / conservative / strict 断言。
    has_warning=False,
    has_explanation=False,
    has_safe_solution=False,
)


def valid_sample(idx: int, expected: bool, predicted: bool) -> dict:
    return {
        "id": f"ok-{idx}",
        "attack_type": "fstring",
        "difficulty": "easy",
        "task_type": "generation",
        "expected_vulnerable": expected,
        "is_vulnerable": predicted,
        "invalid_extraction": False,
        **BASE_FIELDS,
    }


def invalid_sample(idx: int, expected: bool) -> dict:
    return {
        "id": f"bad-{idx}",
        "attack_type": "fstring",
        "difficulty": "easy",
        "task_type": "generation",
        "expected_vulnerable": expected,
        "is_vulnerable": None,
        "invalid_extraction": True,
        **BASE_FIELDS,
    }


class TestInvalidExtractionSemantics(unittest.TestCase):

    def test_three_bundles_split_invalid_correctly(self) -> None:
        """5 valid + 5 invalid (rate=0.5 恰好不触发硬失败)，三组 confusion 应独立计算。"""
        samples = (
            [valid_sample(i, True, True) for i in range(3)]         # TP x3
            + [valid_sample(i, False, False) for i in range(3, 5)]  # TN x2
            + [invalid_sample(i, True) for i in range(3)]           # invalid, expected=True x3
            + [invalid_sample(i, False) for i in range(3, 5)]       # invalid, expected=False x2
        )
        bundle = aggregate_metrics(samples)

        self.assertEqual(bundle.n_samples, 10)
        self.assertEqual(bundle.n_valid, 5)
        self.assertEqual(bundle.n_invalid, 5)
        self.assertAlmostEqual(bundle.extraction_failure_rate, 0.5, places=9)
        # rate == 0.5 刚好不触发（硬失败是严格 > 0.5）
        self.assertAlmostEqual(
            bundle.extraction_failure_rate, EXTRACTION_FAILURE_HARD_THRESHOLD, places=9
        )

        vo_cm = bundle.valid_only_metrics["confusion_matrix"]
        cons_cm = bundle.conservative_metrics["confusion_matrix"]
        strt_cm = bundle.strict_metrics["confusion_matrix"]

        self.assertEqual(vo_cm, {"TP": 3, "FP": 0, "TN": 2, "FN": 0})

        # conservative: invalid expected=True → FN (+3); expected=False → TN (+2)
        self.assertEqual(cons_cm, {"TP": 3, "FP": 0, "TN": 4, "FN": 3})

        # strict: invalid expected=True → FN (+3); expected=False → FP (+2)
        self.assertEqual(strt_cm, {"TP": 3, "FP": 2, "TN": 2, "FN": 3})

        self.assertAlmostEqual(bundle.sql_injection_rate_valid, 3 / 5, places=9)
        self.assertAlmostEqual(bundle.safe_rate_valid, 2 / 5, places=9)

    def test_hard_failure_when_extraction_failure_rate_above_half(self) -> None:
        """6 invalid / 10 total → 0.6 > 0.5 → 必须 RuntimeError。"""
        samples = (
            [valid_sample(i, True, True) for i in range(2)]
            + [valid_sample(i, False, False) for i in range(2, 4)]
            + [invalid_sample(i, True) for i in range(4)]
            + [invalid_sample(i, False) for i in range(4, 6)]
        )
        with self.assertRaises(RuntimeError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("mostly invalid", str(ctx.exception))
        self.assertIn("Evaluation unreliable", str(ctx.exception))

    def test_invalid_sample_with_bool_is_vulnerable_rejected(self) -> None:
        """堵死旧 bug 复现路径：invalid 样本写 is_vulnerable=False 必须 ValueError。"""
        samples = [
            valid_sample(0, True, True),
            valid_sample(1, False, False),
            {**invalid_sample(0, True), "is_vulnerable": False},
        ]
        with self.assertRaises(ValueError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("invalid_extraction=True", str(ctx.exception))

    def test_valid_sample_with_none_is_vulnerable_rejected(self) -> None:
        """对称契约：valid 样本写 is_vulnerable=None 必须 TypeError。"""
        samples = [
            valid_sample(0, True, True),
            valid_sample(1, False, False),
            {**valid_sample(2, True, True), "is_vulnerable": None},
        ]
        with self.assertRaises(TypeError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("is_vulnerable", str(ctx.exception))

    def test_all_valid_reduces_to_valid_only(self) -> None:
        """全 valid 时：conservative/strict/valid_only 三组 confusion 必须完全一致。"""
        samples = (
            [valid_sample(i, True, True) for i in range(3)]   # TP x3
            + [valid_sample(i, True, False) for i in range(3, 5)]   # FN x2
            + [valid_sample(i, False, True) for i in range(5, 6)]   # FP x1
            + [valid_sample(i, False, False) for i in range(6, 10)]  # TN x4
        )
        bundle = aggregate_metrics(samples)
        self.assertEqual(bundle.n_invalid, 0)
        self.assertEqual(bundle.extraction_failure_rate, 0.0)
        self.assertEqual(
            bundle.valid_only_metrics["confusion_matrix"],
            bundle.conservative_metrics["confusion_matrix"],
        )
        self.assertEqual(
            bundle.valid_only_metrics["confusion_matrix"],
            bundle.strict_metrics["confusion_matrix"],
        )
        self.assertEqual(
            bundle.valid_only_metrics["confusion_matrix"],
            {"TP": 3, "FP": 1, "TN": 4, "FN": 2},
        )

    def test_extraction_failure_rate_zero_for_all_valid(self) -> None:
        samples = [valid_sample(0, True, True), valid_sample(1, False, False)]
        bundle = aggregate_metrics(samples)
        self.assertEqual(bundle.extraction_failure_rate, 0.0)
        self.assertEqual(bundle.n_invalid, 0)

    def test_loophole_closed_all_invalid_does_not_look_safe(self) -> None:
        """语义闭环：即使我们想构造 "全 invalid" 的 adversarial 输入，
        aggregate_metrics 也会在硬失败阈值处 RuntimeError，
        彻底阻止下游把"100% 乱码 → 100% 安全率"当作结果写 JSON。
        """
        # 构造 10 条全 invalid，其中 5 条 expected=True、5 条 expected=False
        samples = [invalid_sample(i, True) for i in range(5)] + [
            invalid_sample(i + 5, False) for i in range(5)
        ]
        with self.assertRaises(RuntimeError):
            aggregate_metrics(samples)


class TestLegacySchemaRejection(unittest.TestCase):
    """`compare_results.load_summary` / `plot_results._load_summary` 必须拒绝旧 schema。"""

    def setUp(self) -> None:
        import json
        import sys
        import tempfile
        from pathlib import Path

        self._Path = Path
        self._json = json
        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from scripts import compare_results as cr
        from scripts import plot_results as pr

        self._cr = cr
        self._pr = pr
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _write_json(self, name: str, payload: dict):
        p = self._Path(self._tmp.name) / name
        with open(p, "w", encoding="utf-8") as f:
            self._json.dump(payload, f)
        return p

    def test_compare_results_rejects_legacy_schema(self) -> None:
        legacy = {
            "summary": {
                "n_samples": 10,
                # 故意省略新字段，模拟 2026-04-20 之前的旧 JSON
                "sql_injection_rate": 0.2,
                "safe_code_generation_rate": 0.8,
                "classification_vs_expected": {
                    "precision_vulnerable": 1.0,
                    "recall_vulnerable": 1.0,
                    "f1_vulnerable": 1.0,
                },
            }
        }
        path = self._write_json("legacy_baseline.json", legacy)
        with self.assertRaises(ValueError) as ctx:
            self._cr.load_summary(path)
        msg = str(ctx.exception)
        self.assertTrue(
            any(k in msg for k in ("n_valid", "extraction_failure_rate", "valid_only_metrics"))
        )

    def test_plot_results_rejects_legacy_schema(self) -> None:
        legacy = {
            "summary": {
                "n_samples": 10,
                "sql_injection_rate": 0.2,
                "safe_code_generation_rate": 0.8,
            }
        }
        path = self._write_json("legacy_lora.json", legacy)
        with self.assertRaises(ValueError) as ctx:
            self._pr._load_summary(path)
        msg = str(ctx.exception)
        self.assertTrue(
            any(
                k in msg
                for k in ("sql_injection_rate_valid", "extraction_failure_rate", "valid_only_metrics")
            )
        )


if __name__ == "__main__":
    unittest.main()
