# 2026-04-25 C13：分组指标 invalid_extraction 稀释修复

## 变更目的

修复 `evaluation/metrics.py::_group_rate` 的分母污染问题：旧实现把 `invalid_extraction=True` 样本直接混入分组注入率分母，导致 `by_attack_type_valid` / `by_difficulty_valid` / `by_task_type_valid` 被系统性稀释，无法真实反映模型在“可解析输出”上的性能。

本次改动将分组指标升级为双口径并行输出：

- `valid_only`：仅统计 `invalid_extraction=False`（核心性能口径）
- `all_samples`：保留全量样本（系统级行为观测口径）

## 修改内容

### 1) `evaluation/metrics.py`（Modified）

- `_group_rate` 从单口径改为双口径返回结构：
  - `{"valid_only": {...}, "all_samples": {...}}`
- `aggregate_metrics` 调用改为传入全量 `samples`，由 `_group_rate` 内部切分：
  - `valid_samples = [s for s in samples if not invalid_extraction]`
  - `all_samples = samples`
- 计算逻辑：
  - `rate_valid = vuln(valid_group)/len(valid_group)`
  - `rate_all = vuln(all_group)/len(all_group)`（invalid 只进分母，不进入漏洞分子）
- 安全约束：
  - 无静默 `False` 回退；
  - 空组返回 `None`（避免除零）；
  - 不删除分组指标字段。
- `MetricBundle` 中三项分组字段类型同步更新为嵌套结构类型。
- `explain_metrics()` 文本同步更新，明确双口径语义。

### 2) `tests/test_invalid_extraction_metrics.py`（Modified）

- 新增 `test_group_rates_expose_valid_only_and_all_samples`：
  - 机械验证 `valid_only` 与 `all_samples` 两层都存在；
  - 验证同一组在两种口径下分母不同、数值不同（确认稀释问题已被显式隔离）。

### 3) `README.md`（Modified）

- 新增 “7.1 分组指标双口径（C13）” 说明：
  - 解释 `valid_only` 与 `all_samples` 的区别；
  - 解释为何 invalid_extraction 不能进入核心分组性能口径；
  - 给出 PowerShell 验证命令。

## 影响与结果

- **研究结论更可信**：按攻击类型/难度/任务类型的核心分组对比不再被 invalid 样本冲淡。
- **系统行为仍可观测**：保留 `all_samples` 口径用于监控抽取失败对整体可用性的影响。
- **兼容性变化是显式的**：旧扁平结构已升级为双层结构，调用方需按新 schema 读取，不保留静默兼容路径（符合任务要求）。

## 未改动项

- 未修改训练脚本、模型权重、训练流程。
- 未删除任何历史文件或脚本。
