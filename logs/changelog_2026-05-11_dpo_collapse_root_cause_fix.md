# DPO 训练崩溃根因分析与系统性修复（2026-05-11 第十七次加固）

## 崩溃现象

QLoRA DPO 训练在 step 6-10 发生典型坍缩：

| Step | loss    | grad_norm | entropy   | logps/chosen | logits/chosen |
| ---- | ------- | --------- | --------- | ------------ | ------------- |
| 0    | 9.379   | 12.48     | 9.162     | -900.4       | 15.35         |
| 6    | 5.218   | 16.22     | 9.098     | -876.1       | 15.13         |
| 8    | 2.928   | 7.777     | 8.717     | -851.8       | 14.22         |
| 9    | 2.04    | 0.00018   | 8.572     | -797.5       | 13.94         |
| 10   | 2.14    | **0**     | **7.108** | -1213        | 9.955         |
| 11   | 2.5e-05 | **0**     | **1.305** | -2556        | 1.146         |

坍缩路径: logps 爆炸(-900→-2600) → logits 崩溃(15→1) → entropy 暴跌(9→1.3) → grad_norm 归零

## 根因分析（四层）

### P0: logps 起始值异常 (-900)

正常 SFT 模型对 200-token 序列的 logps 约为 -50 ~ -150。起始值 -900 表示模型为每个 token 分配的平均概率仅为 `exp(-4.5) ≈ 0.011`——几乎随机。

根因：**4bit 量化下的 ref_model 不一致**。旧版 `StableDPOTrainer(ref_model=None)` 让 TRL 内部复制 policy 模型作为 ref。但 4bit 量化模型的复制可能产生不同的权重表示，导致 ref 与 policy 对同一输入产生不同的 logits。DPO loss 公式 `-log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))` 中，初始 policy≈ref 时 loss 应为 0.693，但实测 loss=9.379——说明 ref 模型确实产生了不同的概率分布。

### P1: beta 过小无法约束坍缩

旧版 `beta=0.5` 对模型更新的约束太弱。DPO loss 的梯度正比于 `beta * sigmoid(-beta * margin)`，小 beta 允许模型为了最大化 margin 而产生极端参数更新。beta=5.0 将 KL 约束加强 10 倍，确保模型只能做微调。

### P2: 无熵/梯度监控机制

旧版只有 NaN 检测，没有 entropy 或 logps 的坍塌预警。模型在 step 9 处已发出信号（grad_norm=0.00018），到 step 10 彻底死亡。早停机制应在 entropy 跌破阈值时触发。

### P3: 梯度裁剪不足

旧版 `max_grad_norm=0.5/1.0` 对 DPO 的极端梯度不够。step 1 的 grad_norm=41.06 远超裁剪阈值却仍在更新。`max_grad_norm=0.3` 可有效压制异常梯度。

## 修复方案

### 1. 显式加载独立 ref_model（dpo_train.py / train_qlora_dpo.py）

```python
# 旧版: ref_model=None → TRL内部复制（4bit下不一致）
trainer = StableDPOTrainer(model=model, ref_model=None, ...)

# 新版: 显式从磁盘加载相同adapter到独立模型
ref_model = AutoModelForCausalLM.from_pretrained(base, ...)
ref_model = PeftModel.from_pretrained(ref_model, sft_adapter, is_trainable=False)
for p in ref_model.parameters():
    p.requires_grad = False
trainer = StableDPOTrainer(model=model, ref_model=ref_model, ...)
```

### 2. 超参调整

| 参数                | 旧值     | 新值     | 原因        |
| ------------------- | -------- | -------- | ----------- |
| `dpo.beta`          | 0.5      | **5.0**  | 10× KL 约束 |
| `learning_rate_dpo` | 2e-7     | **5e-8** | 4× 降速     |
| `max_grad_norm`     | 0.5/1.0  | **0.3**  | 严格裁剪    |
| `warmup_ratio`      | 0.03/0.1 | **0.2**  | 更长预热    |

### 3. DPO 坍缩检测回调（DpoCollapseGuardCallback）

新增 `stable_dpo_trainer.py::DpoCollapseGuardCallback`：

| 信号            | 阈值                    | 含义    |
| --------------- | ----------------------- | ------- |
| `entropy < 3.0` | 模型 token 分布完全崩塌 |
| `               | logps/chosen            | > 2000` | log 概率爆炸 |
| `               | logits/chosen           | < 5.0`  | logits 崩溃  |

任一触发 → 立即停止训练 + 保存当前 checkpoint。

## 预期效果

| 指标                 | 修复前         | 修复后（预期）      |
| -------------------- | -------------- | ------------------- |
| 初始 loss            | 9.38 (异常)    | ~0.69 (与 ref 一致) |
| 初始 logps/chosen    | -900 (异常)    | -50 ~ -150 (正常)   |
| entropy 最低值       | 1.2 (完全坍缩) | > 5.0 (正常)        |
| grad_norm 波动       | 0 ~ 41         | 0.01 ~ 0.5 (稳定)   |
| rewards/margins 终值 | +47 (发散)     | 1 ~ 3 (收敛)        |
| 训练崩溃             | step 10        | 不崩溃              |

## 改动的文件

| 文件                                                       | 操作     | 说明                                                                   |
| ---------------------------------------------------------- | -------- | ---------------------------------------------------------------------- |
| `training/dpo_train.py`                                    | 修改     | 显式加载独立 ref_model；beta=5.0；max_grad_norm=0.3；max_prompt_length |
| `training/train_qlora_dpo.py`                              | 修改     | 同上（QLoRA 版本）                                                     |
| `training/stable_dpo_trainer.py`                           | 修改     | 新增 DpoCollapseGuardCallback（熵/logps/logits 三层坍缩检测）          |
| `configs/default.yaml`                                     | 修改     | beta 0.5→5.0；LR 2e-7→5e-8；max_grad_norm 0.5→0.3；warmup 0.1→0.2      |
| `configs/dpo.yaml`                                         | 修改     | beta 0.5→5.0；新增 max_prompt_length                                   |
| `logs/changelog_2026-05-11_dpo_collapse_root_cause_fix.md` | **新建** | 本文件                                                                 |

### 不变更文件
- `dataset/` · `evaluation/` · `detection/` · `tests/` — 零改动
