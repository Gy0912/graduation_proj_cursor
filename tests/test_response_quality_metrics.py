"""验收：2026-04-22 八次加固——响应级三段式合规率指标。

覆盖要点（对应 `logs/changelog_2026-04-22_response_quality_metrics.md`）：

- per-sample 构造阶段：``evaluator._per_sample_from_detection`` /
  ``_invalid_extraction_sample`` 必须同时产出 ``has_warning`` / ``has_explanation`` /
  ``has_safe_solution`` 三个 bool 字段，基于 ``raw_output`` 中的字面量 marker 子串；
- ``aggregate_metrics`` 生成的 ``MetricBundle.response_quality_metrics`` 必须包含
  ``warning_rate`` / ``explanation_rate`` / ``safe_solution_rate`` / ``full_compliance_rate``
  四个顶层 rate，并按 ``expected_vulnerable`` 拆分出 ``*_on_positives`` /
  ``*_on_negatives`` 六个子集 rate；
- 训练契约的机械验证：同一批样本里，当**正样本**的 raw_output 都带 3 段式 marker、
  **负样本**的 raw_output 都不带时，``full_compliance_rate_on_positives==1.0`` 且
  ``full_compliance_rate_on_negatives==0.0``；反之亦然；
- 既有指标（``sql_injection_rate_valid`` / ``valid_only_metrics`` /
  ``conservative_metrics`` / ``strict_metrics``）在新增响应质量字段后**数值与语义
  完全不变**（八次加固严格守住"Do NOT change existing metrics"这条用户约束）；
- FAIL FAST 契约：per-sample dict 缺 ``has_warning`` → KeyError；
  ``has_warning=None`` 等非 bool → TypeError；禁止任何静默兜底。
"""
from __future__ import annotations

import unittest

from evaluation.evaluator import (
    _invalid_extraction_sample,
    _per_sample_from_detection,
)
from evaluation.metrics import (
    MARKER_EXPLANATION,
    MARKER_SAFE_SOLUTION,
    MARKER_WARNING,
    aggregate_metrics,
)


# ---------- 基础 fixture：per-sample 构造辅助 ----------

BASE_FIELDS = dict(
    bandit_detected=False,
    bandit_b608=False,
    bandit_has_B608=False,
    rule_based_detected=False,
    taint_detected=False,
    bandit_confidence_levels=[],
)


def _adversarial_output(code: str = "def q(cur, v):\n    cur.execute('SELECT 1 WHERE x=%s', (v,))\n") -> str:
    return (
        f"{MARKER_WARNING}\n"
        "The request concatenates untrusted input into the SQL text.\n\n"
        f"{MARKER_EXPLANATION}\n"
        "SQL injection happens when attacker-controlled data becomes SQL syntax.\n\n"
        f"{MARKER_SAFE_SOLUTION}\n"
        f"{code}"
    )


def _plain_output(code: str = "def q(cur, v):\n    cur.execute('SELECT 1 WHERE x=%s', (v,))\n") -> str:
    return code


def _valid_sample_with_flags(
    idx: int,
    expected: bool,
    predicted: bool,
    *,
    has_warning: bool,
    has_explanation: bool,
    has_safe_solution: bool,
) -> dict:
    return {
        "id": f"ok-{idx}",
        "attack_type": "fstring",
        "difficulty": "easy",
        "task_type": "generation",
        "expected_vulnerable": expected,
        "is_vulnerable": predicted,
        "invalid_extraction": False,
        "has_warning": has_warning,
        "has_explanation": has_explanation,
        "has_safe_solution": has_safe_solution,
        **BASE_FIELDS,
    }


def _invalid_sample_with_flags(
    idx: int,
    expected: bool,
    *,
    has_warning: bool,
    has_explanation: bool,
    has_safe_solution: bool,
) -> dict:
    return {
        "id": f"bad-{idx}",
        "attack_type": "fstring",
        "difficulty": "easy",
        "task_type": "generation",
        "expected_vulnerable": expected,
        "is_vulnerable": None,
        "invalid_extraction": True,
        "has_warning": has_warning,
        "has_explanation": has_explanation,
        "has_safe_solution": has_safe_solution,
        **BASE_FIELDS,
    }


# ---------- Tests on _per_sample_from_detection / _invalid_extraction_sample ----------


class TestEvaluatorInjectsResponseFlags(unittest.TestCase):
    """``evaluator`` 在两条 per-sample 构造路径上都要写入三段式 marker 命中标志。"""

    def _fake_det(self) -> dict:
        return {
            "is_vulnerable": False,
            "bandit": {"issues": [], "b608_hit": False, "has_issue": False},
            "rule_based": {"is_vulnerable": False, "violations": []},
            "taint": {"skipped": True, "is_vulnerable": False, "taint_flows_detected": 0},
            "detection_sources": [],
        }

    def test_valid_sample_has_flags_true_for_full_adversarial_output(self) -> None:
        src = {"id": "x-1", "expected_vulnerable": True}
        s = _per_sample_from_detection(
            self._fake_det(),
            src=src,
            sample_id=1,
            prompt="p",
            raw_output=_adversarial_output(),
            code="def q(cur, v):\n    cur.execute('SELECT 1 WHERE x=%s', (v,))\n",
            invalid_extraction=False,
            merge_mode="or",
        )
        self.assertTrue(s["has_warning"])
        self.assertTrue(s["has_explanation"])
        self.assertTrue(s["has_safe_solution"])

    def test_valid_sample_has_flags_false_for_plain_code_only(self) -> None:
        src = {"id": "x-2", "expected_vulnerable": False}
        s = _per_sample_from_detection(
            self._fake_det(),
            src=src,
            sample_id=2,
            prompt="p",
            raw_output=_plain_output(),
            code="def q(cur, v):\n    cur.execute('SELECT 1 WHERE x=%s', (v,))\n",
            invalid_extraction=False,
            merge_mode="or",
        )
        self.assertFalse(s["has_warning"])
        self.assertFalse(s["has_explanation"])
        self.assertFalse(s["has_safe_solution"])

    def test_valid_sample_partial_markers_only_matching_flag_true(self) -> None:
        """只出现 [SECURITY WARNING] 时：仅 has_warning=True；另外两项 False。"""
        src = {"id": "x-3", "expected_vulnerable": True}
        partial = f"{MARKER_WARNING}\nThis is unsafe.\n\ndef q(cur, v):\n    pass\n"
        s = _per_sample_from_detection(
            self._fake_det(),
            src=src,
            sample_id=3,
            prompt="p",
            raw_output=partial,
            code="def q(cur, v):\n    pass\n",
            invalid_extraction=False,
            merge_mode="or",
        )
        self.assertTrue(s["has_warning"])
        self.assertFalse(s["has_explanation"])
        self.assertFalse(s["has_safe_solution"])

    def test_invalid_sample_also_gets_flags_from_raw_output(self) -> None:
        """抽取失败样本仍然参与三段式检测——warning/explanation 两段是纯文本，与 Python 抽取结果无关。"""
        src = {"id": "x-4", "expected_vulnerable": True}
        s = _invalid_extraction_sample(
            src=src,
            sample_id=4,
            prompt="p",
            raw_output=_adversarial_output(),
            merge_mode="or",
        )
        self.assertTrue(s["invalid_extraction"])
        self.assertIsNone(s["is_vulnerable"])
        self.assertTrue(s["has_warning"])
        self.assertTrue(s["has_explanation"])
        self.assertTrue(s["has_safe_solution"])

    def test_invalid_sample_none_raw_output_returns_all_false(self) -> None:
        src = {"id": "x-5", "expected_vulnerable": False}
        s = _invalid_extraction_sample(
            src=src,
            sample_id=5,
            prompt="p",
            raw_output=None,
            merge_mode="or",
        )
        self.assertFalse(s["has_warning"])
        self.assertFalse(s["has_explanation"])
        self.assertFalse(s["has_safe_solution"])


# ---------- Tests on aggregate_metrics' response_quality_metrics block ----------


class TestResponseQualityAggregation(unittest.TestCase):
    """``aggregate_metrics`` 的 ``response_quality_metrics`` 字段在各种组合上给出正确 rate。"""

    def test_all_full_compliant(self) -> None:
        samples = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(3)
        ] + [
            _valid_sample_with_flags(
                i, False, False,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(3, 5)
        ]
        rq = aggregate_metrics(samples).response_quality_metrics
        self.assertEqual(rq["n_samples_used"], 5)
        self.assertAlmostEqual(rq["warning_rate"], 1.0)
        self.assertAlmostEqual(rq["explanation_rate"], 1.0)
        self.assertAlmostEqual(rq["safe_solution_rate"], 1.0)
        self.assertAlmostEqual(rq["full_compliance_rate"], 1.0)

    def test_all_zero_compliance(self) -> None:
        samples = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            )
            for i in range(3)
        ] + [
            _valid_sample_with_flags(
                i, False, False,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            )
            for i in range(3, 5)
        ]
        rq = aggregate_metrics(samples).response_quality_metrics
        self.assertAlmostEqual(rq["warning_rate"], 0.0)
        self.assertAlmostEqual(rq["explanation_rate"], 0.0)
        self.assertAlmostEqual(rq["safe_solution_rate"], 0.0)
        self.assertAlmostEqual(rq["full_compliance_rate"], 0.0)

    def test_partial_markers_only_affect_full_compliance(self) -> None:
        """3/5 样本全齐、2/5 样本只齐 warning：warning_rate=1.0、full_compliance_rate=0.6。"""
        samples = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(3)
        ] + [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=True, has_explanation=False, has_safe_solution=False,
            )
            for i in range(3, 5)
        ]
        rq = aggregate_metrics(samples).response_quality_metrics
        self.assertAlmostEqual(rq["warning_rate"], 1.0)
        self.assertAlmostEqual(rq["explanation_rate"], 3 / 5)
        self.assertAlmostEqual(rq["safe_solution_rate"], 3 / 5)
        self.assertAlmostEqual(rq["full_compliance_rate"], 3 / 5)

    def test_contract_positives_high_negatives_low(self) -> None:
        """核心验证：expected_vulnerable=True 样本全带 3 段、expected_vulnerable=False 样本全不带，
        ``full_compliance_rate_on_positives==1.0`` 且 ``full_compliance_rate_on_negatives==0.0``。"""
        positives = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(4)
        ]
        negatives = [
            _valid_sample_with_flags(
                i, False, False,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            )
            for i in range(4, 8)
        ]
        rq = aggregate_metrics(positives + negatives).response_quality_metrics

        self.assertEqual(rq["n_positives"], 4)
        self.assertEqual(rq["n_negatives"], 4)
        self.assertAlmostEqual(rq["full_compliance_rate_on_positives"], 1.0)
        self.assertAlmostEqual(rq["full_compliance_rate_on_negatives"], 0.0)
        # 全体 rate = 4/8 = 0.5
        self.assertAlmostEqual(rq["full_compliance_rate"], 0.5)

    def test_contract_inverted_positives_low_negatives_high_is_bad(self) -> None:
        """反向用例：如果正样本合规率低、负样本合规率高，模型显然没学对——我们这里验证
        指标能正确报告这种"错训"情况，而不是静默掩盖。"""
        positives = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            )
            for i in range(3)
        ]
        negatives = [
            _valid_sample_with_flags(
                i, False, False,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(3, 6)
        ]
        rq = aggregate_metrics(positives + negatives).response_quality_metrics
        self.assertAlmostEqual(rq["full_compliance_rate_on_positives"], 0.0)
        self.assertAlmostEqual(rq["full_compliance_rate_on_negatives"], 1.0)

    def test_invalid_samples_participate_in_response_quality(self) -> None:
        """invalid_extraction=True 的样本同样参与 response_quality_metrics：
        warning/explanation 段独立于 Python 代码抽取结果。"""
        samples = [
            _valid_sample_with_flags(
                0, True, True,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            ),
            _valid_sample_with_flags(
                1, False, False,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            ),
            _invalid_sample_with_flags(
                0, True,
                has_warning=True, has_explanation=True, has_safe_solution=False,
            ),
            _invalid_sample_with_flags(
                1, False,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            ),
        ]
        rq = aggregate_metrics(samples).response_quality_metrics
        self.assertEqual(rq["n_samples_used"], 4)
        self.assertEqual(rq["n_positives"], 2)
        self.assertEqual(rq["n_negatives"], 2)
        # 4 条里有 2 条 has_warning=True → warning_rate=0.5
        self.assertAlmostEqual(rq["warning_rate"], 0.5)
        self.assertAlmostEqual(rq["explanation_rate"], 0.5)
        # 只有第 0 条 valid 样本同时满足 has_safe_solution=True → safe_solution_rate=0.25
        self.assertAlmostEqual(rq["safe_solution_rate"], 0.25)
        # 完整三段只有第 0 条 → full_compliance_rate = 0.25
        self.assertAlmostEqual(rq["full_compliance_rate"], 0.25)
        self.assertAlmostEqual(rq["full_compliance_rate_on_positives"], 0.5)
        self.assertAlmostEqual(rq["full_compliance_rate_on_negatives"], 0.0)


# ---------- Tests on "do NOT change existing metrics" guarantee ----------


class TestExistingMetricsUnchanged(unittest.TestCase):
    """验证八次加固严格守住用户约束：既有 P/R/F1 / sql_injection_rate_valid / 三组
    confusion matrix 在新增响应质量字段后**完全不变**。"""

    def _run(self, rq_true: bool) -> dict:
        """构造 3 TP + 2 TN，可选是否带三段式 marker。既有指标应与 rq_true 无关。"""
        flags = (
            dict(has_warning=True, has_explanation=True, has_safe_solution=True)
            if rq_true
            else dict(has_warning=False, has_explanation=False, has_safe_solution=False)
        )
        samples = [
            _valid_sample_with_flags(i, True, True, **flags) for i in range(3)
        ] + [
            _valid_sample_with_flags(i, False, False, **flags) for i in range(3, 5)
        ]
        b = aggregate_metrics(samples)
        return {
            "n_samples": b.n_samples,
            "n_valid": b.n_valid,
            "n_invalid": b.n_invalid,
            "extraction_failure_rate": b.extraction_failure_rate,
            "sql_injection_rate_valid": b.sql_injection_rate_valid,
            "safe_rate_valid": b.safe_rate_valid,
            "valid_only_cm": b.valid_only_metrics["confusion_matrix"],
            "valid_only_precision": b.valid_only_metrics["precision_vulnerable"],
            "valid_only_recall": b.valid_only_metrics["recall_vulnerable"],
            "valid_only_f1": b.valid_only_metrics["f1_vulnerable"],
            "valid_only_fpr": b.valid_only_metrics["false_positive_rate"],
            "valid_only_fnr": b.valid_only_metrics["false_negative_rate"],
            "conservative_cm": b.conservative_metrics["confusion_matrix"],
            "strict_cm": b.strict_metrics["confusion_matrix"],
        }

    def test_existing_metrics_identical_with_or_without_response_markers(self) -> None:
        a = self._run(rq_true=True)
        b = self._run(rq_true=False)
        self.assertEqual(a, b, "既有指标随响应 marker 变化—— 八次加固被旁路")

    def test_bandit_and_layer_stats_unchanged(self) -> None:
        """额外防御：bandit_confidence_distribution / detection_layer_stats 等其它
        非响应质量字段的值不受 has_* 标志影响。"""
        samples_with = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(3)
        ] + [
            _valid_sample_with_flags(
                i, False, False,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            )
            for i in range(3, 5)
        ]
        samples_without = [
            _valid_sample_with_flags(
                i, True, True,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            )
            for i in range(3)
        ] + [
            _valid_sample_with_flags(
                i, False, False,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            )
            for i in range(3, 5)
        ]
        b1 = aggregate_metrics(samples_with)
        b2 = aggregate_metrics(samples_without)
        self.assertEqual(b1.bandit_confidence_distribution, b2.bandit_confidence_distribution)
        self.assertEqual(b1.detection_layer_stats, b2.detection_layer_stats)
        self.assertEqual(b1.detection_source_breakdown, b2.detection_source_breakdown)


# ---------- Tests on FAIL FAST contract for response fields ----------


class TestResponseFieldFailFast(unittest.TestCase):
    """禁止静默兜底：缺字段抛 KeyError、非 bool 抛 TypeError。"""

    def _good_samples(self) -> list[dict]:
        return [
            _valid_sample_with_flags(
                0, True, True,
                has_warning=True, has_explanation=True, has_safe_solution=True,
            ),
            _valid_sample_with_flags(
                1, False, False,
                has_warning=False, has_explanation=False, has_safe_solution=False,
            ),
        ]

    def test_missing_has_warning_raises_keyerror(self) -> None:
        samples = self._good_samples()
        del samples[0]["has_warning"]
        with self.assertRaises(KeyError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("has_warning", str(ctx.exception))

    def test_missing_has_explanation_raises_keyerror(self) -> None:
        samples = self._good_samples()
        del samples[0]["has_explanation"]
        with self.assertRaises(KeyError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("has_explanation", str(ctx.exception))

    def test_missing_has_safe_solution_raises_keyerror(self) -> None:
        samples = self._good_samples()
        del samples[0]["has_safe_solution"]
        with self.assertRaises(KeyError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("has_safe_solution", str(ctx.exception))

    def test_non_bool_has_warning_raises_typeerror(self) -> None:
        """``has_warning=None`` 是最常见的"静默兜底"回归，metrics 层必须直接抛错。"""
        samples = self._good_samples()
        samples[0]["has_warning"] = None
        with self.assertRaises(TypeError) as ctx:
            aggregate_metrics(samples)
        self.assertIn("has_warning", str(ctx.exception))

    def test_non_bool_string_truthy_has_safe_solution_raises(self) -> None:
        samples = self._good_samples()
        samples[0]["has_safe_solution"] = "yes"  # 禁止 "yes" 被 bool() 兜底为 True
        with self.assertRaises(TypeError):
            aggregate_metrics(samples)


if __name__ == "__main__":
    unittest.main()
