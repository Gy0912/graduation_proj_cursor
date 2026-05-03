# Changelog: 回退本会话（Agent）内全部代码与文档改动

## 说明

按用户要求，**仅撤销本 Cursor Agent 会话中对仓库的修改**，不触碰会话开始前已存在的其它本地未提交改动（如 `data/*`、`dataset/*`、`training/*` 等未列入本会话的变更）。

## 已执行操作

1. **`git restore --source=HEAD`** 恢复以下已跟踪文件至当前分支 **HEAD**（本地最后一次提交）：
   - `README.md`
   - `configs/default.yaml`、`configs/default_run.yaml`、`configs/default_bandit_only_run.yaml`
   - `detection/sql_injection_detector.py`
   - `evaluation/evaluate.py`、`evaluation/evaluator.py`、`evaluation/prompt_loader.py`

2. **删除**本会话新增、且 **HEAD 中不存在** 的无跟踪文件：
   - `evaluation/inference_constraints.py`（仓库历史中无此路径）
   - `tests/test_extraction_robustness.py`
   - `logs/changelog_2026-05-03_eval_prompt_anti_leakage.md`
   - `logs/changelog_2026-05-03_eval_decode_stability.md`
   - `logs/changelog_2026-05-03_extractor_noise_preprocessing.md`

## 影响

- 评测 prompt 防泄漏、贪心解码与 `--raw-preview-only`、抽取器噪声预处理等相关行为均回到 **HEAD** 所对应实现。
- 若本地工作区仍依赖已删除的 `inference_constraints.py`，请改回与 **HEAD** 一致的评测入口（当前 `HEAD` 的 `evaluator.py` 不引用该模块）。
