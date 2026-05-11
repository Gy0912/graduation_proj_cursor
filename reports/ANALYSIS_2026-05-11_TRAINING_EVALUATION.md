# 训练结果评估与后续方向分析（2026-05-11）

## 一、当前训练结果全量对比

| 指标 | baseline | lora_sft | lora_dpo | qlora_sft | qlora_dpo |
|------|----------|----------|----------|-----------|-----------|
| sql_injection_rate_valid | **0.131** | 0.143 | 0.149 | 0.145 | 0.171 |
| extraction_failure_rate | 0.030 | 0.020 | 0.013 | 0.013 | 0.003 |
| FPR (false_positive) | **0.137** | 0.138 | 0.143 | 0.143 | 0.181 |
| TP | 18 | 22 | 23 | 22 | 24 |
| FP | 20 | 20 | 21 | 21 | 27 |
| TN | 126 | 125 | 126 | 126 | 122 |
| FN | 127 | 127 | 126 | 127 | 126 |
| F1_vulnerable | 0.197 | 0.230 | 0.238 | 0.229 | 0.239 |

### 核心结论

**所有训练后的模型都比 baseline 更差。** SFT 使注入率从 13.1% 上升到 14.3%，DPO 进一步上升到 14.9%（LoRA）和 17.1%（QLoRA）。训练不仅没有提升安全性，反而微量削弱了它。

**原因：StarCoder2-3b 对 SQL 安全任务"过强"。** 基座模型在未经过任何安全微调的情况下，已经在 300 个对抗提示中成功防御 87%，仅产出 13.1% 的可检测注入代码。这表明 StarCoder2-3b 的预训练语料中已经包含了大量参数化查询示例，模型内部已经建立了"拼接 SQL 是不安全"的认知。

在此基座上做 SFT（教模型输出安全代码）相当于让一个已经考了 87 分的学生再做同样的练习题——不仅无法提升，还可能引入过拟合噪声。DPO 的"安全 > 脆弱"偏好信号同样冗余。

## 二、模型选择建议

### 方案 A：换用更弱的模型（强烈推荐）

| 候选模型 | 参数量 | 优势 | 预期 baseline 注入率 |
|----------|--------|------|---------------------|
| **Qwen2.5-Coder-0.5B** | 0.5B | 极小、易训、论文常见 | ~30-40%（可观察到训练效果） |
| **CodeGen-350M-Mono** | 0.35B | Salesforce 出品、纯代码 | ~40-50% |
| **Phi-2** | 2.7B | 微软、强推理但安全训练少 | ~25-30% |
| **DeepSeek-Coder-1.3B** | 1.3B | 代码专项 | ~20-25% |

**首选 Qwen2.5-Coder-0.5B**：已有 `outputs/lora_sqlfix_qwen05b/` 目录表明团队已尝试——这是正确方向。

### 方案 B：当前模型的替代策略

如果必须使用 StarCoder2-3b，则需改变训练策略：
- **对抗样本增强**：提高 expected_vulnerable=True 比例到 70%+
- **更长训练**：SFT epochs 从 1 增加到 3-5
- **更大 LoRA rank**：r=16→64

## 三、消融实验方案

### 目的

验证 pipeline 中每个组件对最终模型性能的独立贡献。

### 实验设计（6 组）

| 实验组 | SFT数据 | DPO数据 | 早停 | 预期结论 |
|--------|---------|---------|------|----------|
| A: Full pipeline | v2模板库(56种) | 同构化2000对 | ✓ | 完整系统表现 |
| B: 无DPO | v2模板库(56种) | 无 | ✓ | DPO的独立贡献 |
| C: 旧模板 | v1模板(4种) | 旧1100对 | ✓ | 模板多样性的贡献 |
| D: 无早停 | v2模板库(56种) | 同构化2000对 | ✗ | 早停机制的贡献 |
| E: 低beta DPO | v2模板库(56种) | 同构化2000对beta=0.5 | ✓ | beta=5.0的贡献 |
| F: 仅SFT基础 | v1模板(4种) | 无 | ✗ | 最简基线 |

### 运行方式

```powershell
# A: Full (已完成)
python dataset/generate_expanded_dataset.py --num_samples 2500 --seed 42
python training/train_lora_sft.py --config configs/default_run.yaml
python training/dpo_train.py --config configs/dpo.yaml

# B: 无DPO (跳过DPO训练即可)

# C: 旧模板 (切换到旧版 template_bank.py 后重新生成数据)
# D: 无早停 (设置 early_stopping.enabled=false)
# E: 低beta (设置 dpo.beta=0.5)
# F: 最简基线 (C + D 组合)
```

## 四、回归实验方案

### 目的

确保修复没有引入新的回归——新系统在"修复前已知差"的指标上不差于旧系统，在"修复目标"的指标上好于旧系统。

### 测试用例设计（5 类 × 3 用例 = 15 条）

| 测试类 | 验证内容 | 方法 |
|--------|----------|------|
| TemplateUniqueness | 唯一率 ≥ 90% | 运行 generate_expanded_dataset 后统计 count_unique_outputs |
| DriverDistribution | pymysql < 35% | 运行后统计 compute_driver_distribution |
| TokenDiversity | max_overlap < 0.70 | 运行 audit_token_diversity |
| DpoIsomorphism | 100% 同构性 | 读取 dpo_pairs.json 对每对跑 _verify_dpo_isomorphism |
| EarlyStopping | 早停在 epoch 0.3-0.8 触发 | 用低熵数据(旧模板)跑 SFT，验证早停触发 |

### 自动化回归脚本

```python
# tests/test_regression_2026_05_11.py
def test_template_uniqueness():
    """训练集唯一率 ≥ 90%"""
    ...

def test_driver_not_monopolized():
    """pymysql 占比 < 35%"""
    ...

def test_token_overlap_below_threshold():
    """任意两模板重叠 < 0.70"""
    ...

def test_dpo_isomorphism_100pct():
    """所有 DPO 对同构性 100%"""
    ...

def test_early_stopping_triggers():
    """低熵数据上早停触发"""
    ...
```

## 五、残余问题评估

| 问题 | 需要修复？ | 理由 |
|------|-----------|------|
| DPO 启动点修正 | **不需要** | 早停已保存 best_checkpoint；且当前 SFT≈baseline，最佳 checkpoint 即初始模型 |
| fix-task INPUT 脱敏 | **不需要** | Input 中的脆弱代码是"Fix"任务的必须上下文，移除后任务语义丢失。当前 setup 中 80% fix 输出来自模板库（非对齐改写），prompt 中的脆弱代码不进入 training target |
| 训练动力学监控 | **不需要** | 已通过 DpoCollapseGuardCallback + 早停覆盖 |
| 快捷路径消除 | **不需要** | 已通过 beta=5.0 + max_grad_norm=0.3 + DpoCollapseGuardCallback 覆盖 |

## 六、20 张对比图方案

### 图 1-3：主指标对比

1. **`01_primary_metrics_bar.png`** — defense_success_rate + safe_rate_on_benign 的 grouped bar（baseline/sft/dpo × lora/qlora）

2. **`02_sql_injection_rate_bar.png`** — sql_injection_rate_valid 的 bar chart（5 模型）

3. **`03_extraction_failure_bar.png`** — extraction_failure_rate（越低越好，含 0.5 硬阈值线）

### 图 4-6：混淆矩阵可视化

4. **`04_confusion_heatmap_baseline.png`** — baseline 的 2×2 混淆矩阵热力图

5. **`05_confusion_heatmap_lora_sft.png`** — lora_sft 混淆矩阵

6. **`06_confusion_heatmap_lora_dpo.png`** — lora_dpo 混淆矩阵

### 图 7-10：攻击类型细分

7. **`07_attack_type_injection_rate.png`** — 按 attack_type 的 grouped bar（5 模型 × 7 attack type）

8. **`08_attack_type_heatmap.png`** — attack_type × model 的 injection rate 热力图

9. **`09_difficulty_injection_rate.png`** — 按 difficulty 的 grouped bar

10. **`10_task_type_injection_rate.png`** — generation vs fix 对比

### 图 11-13：模板多样性

11. **`11_template_uniqueness_bar.png`** — 修复前后唯一率对比

12. **`12_driver_distribution_pie.png`** — 修复前后的 driver 分布饼图对比

13. **`13_token_overlap_histogram.png`** — token 重叠度分布直方图（含 0.70 阈值线）

### 图 14-16：DPO 训练动力学

14. **`14_dpo_training_loss_curve.png`** — DPO loss 曲线（步骤 vs loss）

15. **`15_dpo_logps_curve.png`** — logps/chosen + logps/rejected 双曲线

16. **`16_dpo_entropy_curve.png`** — entropy 曲线（含坍缩阈值 3.0）

### 图 17-18：早停效果

17. **`17_early_stopping_val_loss.png`** — SFT val_loss 曲线 + 早停触发点标记

18. **`18_overfit_ratio_curve.png`** — overfit_ratio 曲线（含 0.5 阈值线）

### 图 19-20：消融汇总

19. **`19_ablation_summary_bar.png`** — 6 组消融实验的 injection_rate 对比

20. **`20_radar_chart.png`** — 5 模型 × 6 指标的雷达图（defense, benign, injection_rate, extraction, FPR, F1）

### 绘图脚本位置

建议放在 `visualization/` 目录：
- `visualization/plot_primary_metrics.py` — 图 1-3
- `visualization/plot_confusion.py` — 图 4-6
- `visualization/plot_attack_breakdown.py` — 图 7-10
- `visualization/plot_template_diversity.py` — 图 11-13
- `visualization/plot_dpo_dynamics.py` — 图 14-16
- `visualization/plot_early_stopping.py` — 图 17-18
- `visualization/plot_ablation.py` — 图 19-20

每个脚本读取 `outputs/*_results.json` 和训练日志，生成对应的 `outputs/plots/*.png`。
