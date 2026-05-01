# 2026-04-25 退化基线排除（LoRA-only / QLoRA-only）

## 背景与目的

`lora_only` 与 `qlora_only` 在当前训练设定下属于退化基线：LoRA 权重采用零初始化，行为与 `baseline` 数学上等价；QLoRA-only 的差异仅来源于量化噪声，不构成有意义的方法改进。继续把它们当成独立方法参与对比，会污染改进率结论。

本次更新目标：只保留真实方法差异（`baseline`、`lora_sft`、`lora_dpo`、`qlora_sft`、`qlora_dpo`），并在比较前自动剔除“输出完全相同”的重复模型。

## 变更清单

### 修改（Modified）

1. `scripts/compare_results.py`
   - 移除比较方法中的 `lora_only` / `qlora_only`，`METHODS` 只保留 5 个有效方法。
   - 不再为退化方法计算任何指标与 `vs baseline` 改进率。
   - 对配置中的 `lora_only_results` / `qlora_only_results` 给出跳过告警，避免误纳入。
   - 新增重复输出检测：若两个模型逐样本 `raw_output`、`code`、`is_vulnerable` 全部一致，发出
     `Degenerate model detected (identical outputs)` 警告，并将重复模型排除出最终比较。
   - 在写出 `comparison_summary` 前同步清理被排除模型的顶层指标键，防止下游读取到误导字段。

2. `README.md`
   - 更新评测与对比说明，仅展示 5 个有效方法。
   - 补充退化基线排除理由与方法学解释，明确该排除是为了保证比较结论只反映真实方法差异。

## 影响评估

- **结论可信度提升**：比较表和改进率不再被“与 baseline 等价”的方法稀释或误导。
- **鲁棒性提升**：即使未来出现新的退化模型（输出与其他模型完全一致），也会在汇总阶段被自动警告并剔除。
- **兼容性保持**：训练脚本与模型权重均未改动，未触碰训练流程，仅收紧比较与汇总口径。

## 明确未做（Not Changed）

- 未删除任何训练脚本（含 `train_lora_only.py`、`train_qlora_only.py`）。
- 未修改任何模型权重。
- 未触发任何重训练流程。

## 文档说明（对外口径）

LoRA-only and QLoRA-only models are excluded from evaluation because they are mathematically equivalent to the baseline model (LoRA weights are zero-initialized). Any observed differences are due to numerical noise and do not reflect meaningful model behavior.
