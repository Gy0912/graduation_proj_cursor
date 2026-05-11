"""
回归测试套件（2026-05-11）。

验证所有修复未引入回归，各项指标满足基线要求。
运行: pytest tests/test_regression_2026_05_11.py -v
"""
import ast
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _load_dpo_pairs():
    """加载 DPO 对（懒加载）。"""
    path = ROOT / "data" / "dpo_pairs.json"
    if not path.exists():
        return []
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def _load_eval_json(mode: str) -> dict | None:
    path = ROOT / "outputs" / f"{mode}_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


# ═══════════════════════════════════════════════════════════════
# 测试类 1：模板多样性
# ═══════════════════════════════════════════════════════════════

class TestTemplateDiversity:
    """验证模板库满足唯一率和 token 重叠度要求。"""

    @pytest.mark.skipif(
        not (ROOT / "dataset" / "template_bank.py").exists(),
        reason="template_bank.py not found",
    )
    def test_template_count_ge_55(self):
        """模板总数 ≥ 55。"""
        from dataset.template_bank import _TEMPLATES
        assert len(_TEMPLATES) >= 55, f"Only {len(_TEMPLATES)} templates"

    @pytest.mark.skipif(
        not (ROOT / "dataset" / "template_bank.py").exists(),
        reason="template_bank.py not found",
    )
    def test_all_templates_ast_valid(self):
        """所有模板 AST 可解析且不含脆弱 SQL 模式。"""
        from dataset.template_bank import _TEMPLATES
        from dataset.adversarial import contains_vulnerable_sql_pattern
        for t in _TEMPLATES:
            code = t["template"].format(func="fn", param="p", table="tbl", col="col")
            ast.parse(code)
            vuln, names = contains_vulnerable_sql_pattern(code)
            assert not vuln, f"Template {t['idx']} hits patterns: {names}"

    @pytest.mark.skipif(
        not (ROOT / "dataset" / "template_bank.py").exists(),
        reason="template_bank.py not found",
    )
    def test_token_overlap_below_70pct_real(self):
        """实际使用中最大 token 重叠度 < 0.70。"""
        from dataset.template_bank import (
            TemplateSampler, audit_token_diversity, _TEMPLATE_MARKER
        )
        sampler = TemplateSampler(random.Random(42))
        codes = []
        for i in range(56):
            code, _, _ = sampler.sample_template()
            codes.append(code)
        audit = audit_token_diversity(codes, max_pairs=2000)
        assert audit["max_pairwise_overlap"] < 0.70, (
            f"Max overlap {audit['max_pairwise_overlap']} >= 0.70"
        )

    @pytest.mark.skipif(
        not (ROOT / "data" / "train_expanded.json").exists(),
        reason="train_expanded.json not found",
    )
    def test_training_uniqueness_ge_70pct(self):
        """训练集唯一率 ≥ 70%（真实流水线生成）。"""
        from dataset.template_bank import count_unique_outputs
        with open(ROOT / "data" / "train_expanded.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        outputs = [r["output"] for r in data if "output" in r]
        uq = count_unique_outputs(outputs)
        assert uq["uniqueness_pct"] >= 70.0, (
            f"Uniqueness {uq['uniqueness_pct']:.1f}% < 70%"
        )


# ═══════════════════════════════════════════════════════════════
# 测试类 2：Driver 分布
# ═══════════════════════════════════════════════════════════════

class TestDriverDistribution:
    """验证 driver 分布均衡。"""

    @pytest.mark.skipif(
        not (ROOT / "data" / "train_expanded.json").exists(),
        reason="train_expanded.json not found",
    )
    def test_pymysql_below_40pct(self):
        """pymysql 占比 < 40%。"""
        from dataset.template_bank import compute_driver_distribution
        with open(ROOT / "data" / "train_expanded.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        outputs = [r["output"] for r in data if "output" in r]
        dist = compute_driver_distribution(outputs)
        pymysql_pct = dist.get("pymysql", 0) * 100
        assert pymysql_pct < 40.0, f"pymysql {pymysql_pct:.1f}% >= 40%"

    @pytest.mark.skipif(
        not (ROOT / "data" / "train_expanded.json").exists(),
        reason="train_expanded.json not found",
    )
    def test_no_driver_above_40pct(self):
        """没有任何 driver 占比 > 40%。"""
        from dataset.template_bank import compute_driver_distribution
        with open(ROOT / "data" / "train_expanded.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        outputs = [r["output"] for r in data if "output" in r]
        dist = compute_driver_distribution(outputs)
        for drv, pct in dist.items():
            assert pct * 100 <= 40.0, f"{drv} {pct*100:.1f}% > 40%"


# ═══════════════════════════════════════════════════════════════
# 测试类 3：DPO 同构性
# ═══════════════════════════════════════════════════════════════

class TestDpoIsomorphism:
    """验证 DPO 对 chosen/rejected 同构性。"""

    @pytest.fixture(autouse=False)
    def dpo_pairs(self):
        return _load_dpo_pairs()

    @pytest.mark.skipif(
        not (ROOT / "data" / "dpo_pairs.json").exists(),
        reason="dpo_pairs.json not found",
    )
    def test_dpo_pair_count_ge_1500(self):
        """DPO 对数量 ≥ 1500。"""
        pairs = _load_dpo_pairs()
        assert len(pairs) >= 1500, f"Only {len(pairs)} pairs"

    @pytest.mark.skipif(
        not (ROOT / "data" / "dpo_pairs.json").exists(),
        reason="dpo_pairs.json not found",
    )
    def test_dpo_isomorphism_100pct(self):
        """所有 DPO 对同构性 100%。"""
        from dataset.generate_expanded_dataset import _verify_dpo_isomorphism
        pairs = _load_dpo_pairs()
        failures = []
        for i, p in enumerate(pairs):
            ok, reason = _verify_dpo_isomorphism(p["chosen"], p["rejected"])
            if not ok:
                failures.append((i, reason))
        assert len(failures) == 0, (
            f"{len(failures)}/{len(pairs)} pairs fail isomorphism: "
            f"{failures[:5]}"
        )

    @pytest.mark.skipif(
        not (ROOT / "data" / "dpo_pairs.json").exists(),
        reason="dpo_pairs.json not found",
    )
    def test_dpo_tiers_exist(self):
        """DPO 对包含 easy/medium/hard 三层。"""
        pairs = _load_dpo_pairs()
        tiers = set(p.get("dpo_difficulty_tier", "") for p in pairs)
        assert "easy" in tiers, "Missing easy tier"
        assert "medium" in tiers, "Missing medium tier"
        assert "hard" in tiers, "Missing hard tier"


# ═══════════════════════════════════════════════════════════════
# 测试类 4：评测结果一致性
# ═══════════════════════════════════════════════════════════════

class TestEvaluationConsistency:
    """验证评测结果 schema 完整性和数值合理性。"""

    @pytest.mark.parametrize("model", ["baseline", "lora_sft", "lora_dpo"])
    def test_eval_summary_has_required_fields(self, model):
        """评测 summary 包含所有必要字段。"""
        data = _load_eval_json(model)
        if data is None:
            pytest.skip(f"{model}_results.json not found")
        s = data["summary"]
        required = [
            "n_samples", "n_valid", "n_invalid",
            "extraction_failure_rate",
            "sql_injection_rate_valid",
            "defense_success_rate",
            "safe_rate_on_benign",
            "valid_only_metrics",
            "response_quality_metrics",
        ]
        for field in required:
            assert field in s, f"Missing {field} in {model} summary"

    @pytest.mark.parametrize("model", ["baseline", "lora_sft", "lora_dpo"])
    def test_extraction_failure_rate_below_10pct(self, model):
        """extraction_failure_rate < 0.10。"""
        data = _load_eval_json(model)
        if data is None:
            pytest.skip(f"{model}_results.json not found")
        rate = data["summary"]["extraction_failure_rate"]
        assert rate < 0.10, f"{model} extraction_failure_rate={rate}"


# ═══════════════════════════════════════════════════════════════
# 测试类 5：配置一致性
# ═══════════════════════════════════════════════════════════════

class TestConfigConsistency:
    """验证配置文件内部一致性。"""

    def test_dpo_beta_is_5(self):
        """DPO beta 应为 5.0。"""
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text("utf-8"))
        beta = cfg.get("dpo", {}).get("beta", 0)
        assert beta == 5.0, f"DPO beta={beta}, expected 5.0"

    def test_max_grad_norm_is_03(self):
        """max_grad_norm 应为 0.3。"""
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text("utf-8"))
        mgn = cfg.get("training", {}).get("max_grad_norm", 0)
        assert mgn == 0.3, f"max_grad_norm={mgn}, expected 0.3"

    def test_dpo_lr_is_5e8(self):
        """DPO learning_rate 应为 5e-8。"""
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text("utf-8"))
        lr = cfg.get("training", {}).get("learning_rate_dpo", 0)
        assert lr == 5.0e-8, f"learning_rate_dpo={lr}, expected 5e-8"


# ═══════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
