"""
消融实验运行脚本（2026-05-11）。

运行 6 组消融实验，每组变体通过独立的 config 文件与默认配置合并。
每组实验：生成数据 → 训练 SFT → (可选) 训练 DPO → 评测 → 收集指标。

用法：
  python scripts/run_ablation.py --groups A B C D E F
  python scripts/run_ablation.py --groups A B           # 仅运行 A+B
  python scripts/run_ablation.py --groups all --dry-run # 预览命令
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════
# 实验组定义
# ═══════════════════════════════════════════════════════════════

ABLATION_GROUPS: dict[str, dict] = {
    "A": {
        "name": "Full Pipeline",
        "config": "configs/ablation/a_full.yaml",
        "description": "v2模板(56种) + DPO 2000对(isomorphic) + 早停 + beta=5.0",
        "run_dpo": True,
    },
    "B": {
        "name": "No DPO",
        "config": "configs/ablation/b_no_dpo.yaml",
        "description": "仅SFT，无DPO训练",
        "run_dpo": False,
    },
    "C": {
        "name": "Old Templates",
        "config": "configs/ablation/c_old_templates.yaml",
        "description": "模拟旧版4模板(v1风格)，DPO+早停+b5",
        "run_dpo": True,
        "template_mode": "basic",
    },
    "D": {
        "name": "No Early Stop",
        "config": "configs/ablation/d_no_early_stop.yaml",
        "description": "禁用早停，完整1 epoch训练",
        "run_dpo": True,
    },
    "E": {
        "name": "Low Beta DPO",
        "config": "configs/ablation/e_low_beta.yaml",
        "description": "DPO beta=0.5 (vs 5.0)",
        "run_dpo": True,
    },
    "F": {
        "name": "Minimal Baseline",
        "config": "configs/ablation/f_minimal.yaml",
        "description": "旧模板 + 无DPO + 无早停",
        "run_dpo": False,
        "template_mode": "basic",
    },
}

# 输出路径模板
OUTPUT_DIR = ROOT / "outputs" / "ablation"
EVAL_OUTPUT_TEMPLATE = "outputs/ablation/group_{group}_{model}_results.json"
RESULTS_JSON = ROOT / "outputs" / "ablation" / "_all_results.json"


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _merge_configs(base_path: str, override_path: str | None) -> dict:
    """合并两个 YAML 配置（后者覆盖前者）。"""
    with open(ROOT / base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override_path:
        ov_path = ROOT / override_path
        if ov_path.exists():
            with open(ov_path, "r", encoding="utf-8") as f:
                ov = yaml.safe_load(f)
            _deep_merge(cfg, ov)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    """原地深度合并 override 到 base。"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def run(cmd: list[str], allow_fail: bool = False) -> int:
    """运行命令，返回退出码。"""
    print(f"  >> {' '.join(cmd)}")
    r = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if r.returncode != 0 and not allow_fail:
        print(f"  !! FAILED (exit {r.returncode})")
    return r.returncode


def _run_data_generation(
    group_id: str, py: str, template_mode: str | None = None
) -> int:
    """生成数据集。对 basic 模式传入环境变量限制模板类型。"""
    cmd = [py, "dataset/generate_expanded_dataset.py",
           "--num_samples", "2500", "--eval_ratio", "0.12", "--seed", "42"]
    env = os.environ.copy()
    if template_mode:
        env["ABLATION_TEMPLATE_MODE"] = template_mode
    print(f"  >> (ABLATION_TEMPLATE_MODE={template_mode}) {' '.join(cmd)}")
    r = subprocess.run(cmd, check=False, cwd=str(ROOT), env=env)
    return r.returncode


def run_ablation_group(
    group_id: str,
    group_info: dict,
    py: str,
    dry_run: bool = False,
) -> dict:
    """运行单个消融实验组。

    Returns:
        dict with keys: group, status, metrics, error
    """
    result = {"group": group_id, "name": group_info["name"], "status": "unknown"}
    cfg = _merge_configs("configs/default.yaml", group_info["config"])
    dcfg = _merge_configs("configs/default.yaml", "configs/dpo.yaml")
    if group_info["config"]:
        ov_path = ROOT / group_info["config"]
        if ov_path.exists():
            with open(ov_path, "r", encoding="utf-8") as f:
                _deep_merge(dcfg, yaml.safe_load(f))

    print(f"\n{'='*60}")
    print(f"  Ablation {group_id}: {group_info['name']}")
    print(f"  {group_info['description']}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] skipping actual execution")
        result["status"] = "dry_run"
        return result

    # Step 1: 生成数据
    print(f"\n  [1/4] Generating dataset...")
    template_mode = group_info.get("template_mode")
    rc = _run_data_generation(group_id, py, template_mode)
    if rc != 0:
        result["status"] = "data_gen_failed"
        return result
    run([py, "scripts/build_eval_fixed.py"])

    # Step 2: 训练 SFT
    print(f"\n  [2/4] Training LoRA SFT...")
    rc = run([py, "training/train_lora_sft.py",
              "--config", group_info["config"]], allow_fail=True)
    if rc != 0:
        result["status"] = "sft_failed"
        return result

    # Step 3: (可选) 训练 DPO
    if group_info.get("run_dpo", True):
        print(f"\n  [3/4] Training LoRA DPO...")
        rc = run([py, "training/dpo_train.py",
                  "--config", group_info["config"]], allow_fail=True)
        if rc != 0:
            print(f"  !! DPO training failed (non-fatal, continuing with SFT only)")

    # Step 4: 评测
    print(f"\n  [4/4] Evaluating...")
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 评测 SFT
    sft_out = out_dir / f"group_{group_id}_sft_results.json"
    run([py, "evaluation/evaluate.py",
         "--config", group_info["config"],
         "--model", "lora_sft",
         "--output", str(sft_out)], allow_fail=True)

    # 评测 DPO
    if group_info.get("run_dpo", True):
        dpo_out = out_dir / f"group_{group_id}_dpo_results.json"
        run([py, "evaluation/evaluate.py",
             "--config", group_info["config"],
             "--model", "lora_dpo",
             "--output", str(dpo_out)], allow_fail=True)

    # 收集指标
    result["status"] = "completed"
    result["sft_metrics"] = _load_summary_if_exists(sft_out)
    result["dpo_metrics"] = _load_summary_if_exists(
        out_dir / f"group_{group_id}_dpo_results.json"
    )
    return result


def _load_summary_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
        s = data.get("summary", {})
        return {
            "n_samples": s.get("n_samples"),
            "n_valid": s.get("n_valid"),
            "n_invalid": s.get("n_invalid"),
            "extraction_failure_rate": s.get("extraction_failure_rate"),
            "sql_injection_rate_valid": s.get("sql_injection_rate_valid"),
            "defense_success_rate": s.get("defense_success_rate"),
            "safe_rate_on_benign": s.get("safe_rate_on_benign"),
        }
    except Exception:
        return None


def _fmt_metric(m: dict, key: str) -> str:
    """格式化指标值：None/缺失 → 'N/A'，数值 → .4f 格式。"""
    v = m.get(key)
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def generate_ablation_report(all_results: list[dict]) -> str:
    """生成消融实验 Markdown 报告。"""
    lines = [
        "# 消融实验报告",
        f"\n生成时间: {datetime.now(timezone.utc).isoformat()}\n",
        "\n## 实验结果汇总\n",
        "| Group | Description | Model | inj_valid | defense | benign | ext_fail | Status |",
        "|-------|-------------|-------|-----------|---------|--------|----------|--------|",
    ]

    for r in all_results:
        desc = ABLATION_GROUPS.get(r["group"], {}).get("description", "")[:50]
        for model_key in ["sft_metrics", "dpo_metrics"]:
            m = r.get(model_key)
            if m is None:
                continue
            model_name = "SFT" if model_key == "sft_metrics" else "DPO"
            lines.append(
                f"| {r['group']}: {r['name']} | {desc} | {model_name} | "
                f"{_fmt_metric(m, 'sql_injection_rate_valid')} | "
                f"{_fmt_metric(m, 'defense_success_rate')} | "
                f"{_fmt_metric(m, 'safe_rate_on_benign')} | "
                f"{_fmt_metric(m, 'extraction_failure_rate')} | "
                f"{r['status']} |"
            )

    lines.append("\n## 结论\n")
    lines.append("（需根据实际运行结果填写）\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument(
        "--groups", nargs="+", default=["A"],
        help="要运行的实验组 (A/B/C/D/E/F 或 all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印要执行的命令，不实际运行"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="仅从已有结果生成报告"
    )
    args = parser.parse_args()

    if args.groups == ["all"]:
        groups_to_run = list(ABLATION_GROUPS.keys())
    else:
        groups_to_run = [g.upper() for g in args.groups]

    py = sys.executable

    if args.report:
        # 仅生成报告
        if RESULTS_JSON.exists():
            all_results = json.loads(RESULTS_JSON.read_text("utf-8"))
            report = generate_ablation_report(all_results)
            report_path = OUTPUT_DIR / "ablation_report.md"
            report_path.write_text(report, "utf-8")
            print(f"Report written to {report_path}")
        return

    all_results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for group_id in groups_to_run:
        if group_id not in ABLATION_GROUPS:
            print(f"Unknown group: {group_id}. Valid: {list(ABLATION_GROUPS.keys())}")
            continue

        result = run_ablation_group(
            group_id, ABLATION_GROUPS[group_id], py, dry_run=args.dry_run
        )
        all_results.append(result)

    # 保存结果
    if not args.dry_run:
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nAll results saved to {RESULTS_JSON}")

        report = generate_ablation_report(all_results)
        report_path = OUTPUT_DIR / "ablation_report.md"
        report_path.write_text(report, "utf-8")
        print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
