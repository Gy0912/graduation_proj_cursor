# scripts/

## 脚本清单（当前有效）

| 脚本 | 作用 |
|------|------|
| `00_prepare_env.ps1` | Windows 下创建虚拟环境并安装依赖 |
| `build_dataset.py` | 仅生成 `dataset/*.jsonl` 的 SFT/DPO demo；**不再覆盖评测集** |
| `build_eval_fixed.py` | **合并 `data/generation/eval.json` + `data/fix/eval.json` → `data/combined/eval_fixed.json`；强 schema 校验** |
| `compare_results.py` | 汇总 7 组实验对比 → `outputs/comparison_summary.json` |
| `migrate_dataset_to_research_schema.py` | 将扩展数据迁移到 `data/combined` 研究 schema；评测行强制带 `expected_vulnerable` |
| `plot_results.py` | 从各模型结果 JSON 画 SQL 注入率与 FPR/FNR 图 |
| `prepare_default_run.py` | 从 `configs/default.yaml` 生成 `configs/default_run.yaml` |
| `prepare_bandit_only_run.py` | 从 `default_run.yaml` 生成 `default_bandit_only_run.yaml` |
| `run_thesis_pipeline.py` | 顺序跑全流程（含 `build_eval_fixed.py`，支持 `--skip-lora-dpo/--skip-qlora-dpo`） |

统一术语：训练入口在 `training/`，评测入口在 `evaluation/evaluate.py`，本目录主要提供准备与编排脚本。
