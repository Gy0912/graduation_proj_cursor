# SFT 早停机制与过拟合检测（2026-05-10 第十一次加固）

## 问题诊断

### 症状

SFT 训练在 1 epoch 内 loss 急速下降至接近 0（低熵数据导致模型快速记忆模板），但训练没有任何过拟合检测或拦截机制，导致：

1. 模型在「完全记忆模板」的状态下进入 DPO 阶段
2. 过拟合的 SFT checkpoint 使 DPO 信号噪声比急剧恶化
3. 级联崩溃：SFT 过拟合 → DPO log-ratio 爆炸 → NaN → collapse

### 根因分析

```
当前训练流程
    trainer.train()  ──→  无 val_loss 监控  ──→  最终 checkpoint = 过拟合 checkpoint
         │                                              │
         └─ loss→0, 无停止机制                           └─ DPO 从此启动 → NaN collapse

期望流程
    trainer.train()  ──→  每 N 步 eval val_loss
         │                      │
         ├─ val_loss 改善 → 保存 best_checkpoint
         ├─ val_loss 恶化连续 patience 步 → 早停
         └─ overfit_ratio < 0.5 → 打印过拟合警告
                                     │
                              DPO 从 best_checkpoint 启动 → 正常训练
```

HuggingFace Trainer / TRL 虽然支持 `eval_steps` + `eval_strategy="steps"`，但**不内置**：
- 基于 val_loss 的早停（`EarlyStoppingCallback` 虽存在于 transformers 但仅支持 metric，需手动计算 val_loss）
- overfit_ratio 监控
- 最佳 checkpoint 自动标记供下游（DPO）消费

## 修复方案

### 1. 新增 `training/early_stopping.py`

**`EarlyStoppingCallback`**（`TrainerCallback` 子类）：

| 参数                     | 默认值 | 说明                                                |
| ------------------------ | ------ | --------------------------------------------------- |
| `patience`               | 5      | 连续 val_loss 未改善步数后停止                      |
| `min_delta`              | 1e-4   | 改善的最小绝对变化量                                |
| `overfit_warn_threshold` | 0.5    | train_loss/val_loss 低于此值触发过拟合警告          |
| `overfit_warn_patience`  | 3      | 连续恶化步数才打印过拟合警告（抑制噪声）            |
| `save_best`              | true   | 每次 val_loss 创历史新低时保存到 `best_checkpoint/` |

**`on_evaluate` 行为：**
1. 提取 `eval_loss` 和最近 `train_loss`
2. 计算 `overfit_ratio = train_loss / val_loss`
3. 若 `overfit_ratio < 0.5` 且连续恶化 → 打印过拟合警告
4. 若 `val_loss < best_val_loss - min_delta` → 更新最佳值、重置 patience 计数
5. 若连续 `patience` 步未改善 → 设置 `control.should_training_stop = True`
6. 每次 eval 打印一行状态日志

**`on_save` 行为：**
- 当 trainer 保存 checkpoint 且该 checkpoint 对应最佳步数时，复制 adapter 权重到 `best_checkpoint/`

**`resolve_best_sft_checkpoint(sft_output_dir)`：**
- DPO 侧入口函数
- 读取 `best_checkpoint.json` marker → 若 `best_checkpoint/` 目录非空 → 返回最佳 checkpoint 路径
- 否则回退到最终 checkpoint（保持向后兼容）

**`print_early_stop_summary(callback)`：**
- 训练结束后的审计摘要：状态（早停/完成）、最佳 val_loss、最终 val_loss、退化幅度、过拟合警告、最近历史

### 2. 修改 SFT 训练入口

**`training/train_lora_sft.py`**：
- 导入 `EarlyStoppingCallback`、`print_early_stop_summary`
- 从 `training.early_stopping` 段读取配置参数
- 将 `early_stop_cb` 加入 `SFTTrainer` 的 `callbacks` 列表
- 训练后在 `best_checkpoint/` 子目录独立保存最佳 adapter
- 打印早停摘要

**`training/train_qlora_sft.py`**：
- 同上（QLoRA 版本）

### 3. 修改 DPO 训练入口

**`training/dpo_train.py`**：
- 导入 `resolve_best_sft_checkpoint`
- SFT adapter 加载前调用 `resolve_best_sft_checkpoint(sft_dir)` → 优先从 `best_checkpoint/` 加载
- 打印加载来源（最佳 checkpoint vs 最终 checkpoint）

**`training/train_qlora_dpo.py`**：
- 同上（QLoRA 版本）

### 4. 配置更新

**`configs/default.yaml`** 新增 `training.early_stopping` 段：
```yaml
early_stopping:
  patience: 5
  min_delta: 1.0e-4
  overfit_warn_threshold: 0.5
  overfit_warn_patience: 3
  save_best: true
```

## 改动的文件

| 文件                                          | 操作     | 说明                                                                           |
| --------------------------------------------- | -------- | ------------------------------------------------------------------------------ |
| `training/early_stopping.py`                  | **新建** | EarlyStoppingCallback + resolve_best_sft_checkpoint + print_early_stop_summary |
| `training/train_lora_sft.py`                  | 修改     | 集成 EarlyStoppingCallback；训练后在 best_checkpoint/ 独立保存                 |
| `training/train_qlora_sft.py`                 | 修改     | 同上（QLoRA 版本）                                                             |
| `training/dpo_train.py`                       | 修改     | DPO 从 resolve_best_sft_checkpoint() 加载（最佳 > 最终）                       |
| `training/train_qlora_dpo.py`                 | 修改     | 同上（QLoRA 版本）                                                             |
| `configs/default.yaml`                        | 修改     | 新增 training.early_stopping 配置段                                            |
| `logs/changelog_2026-05-10_early_stopping.md` | **新建** | 本文件                                                                         |

### 不变更文件
- `detection/`：检测逻辑零改动
- `evaluation/`：评测管线零改动
- `dataset/`：数据生成零改动
- `tests/`：回归测试套件零改动
- `training/sft_preprocess.py`：预处理零改动
- `training/stable_dpo_trainer.py`：DPO trainer 零改动

## 验证方式

| 验证步骤                                          | 通过标准                                                |
| ------------------------------------------------- | ------------------------------------------------------- |
| 在低熵原训练集（唯一率 25.7%）上运行 SFT          | 早停在 epoch 0.5-0.8 左右触发                           |
| 确认早停 checkpoint 的 val_loss < 最终 checkpoint | val_loss 差值 >0.05                                     |
| 从早停 checkpoint 启动 DPO                        | DPO loss 正常下降，无 NaN                               |
| 在修复后高熵训练集（唯一率 ≥90%）上运行 SFT       | 早停不触发（或触发在接近 1 epoch 处），模型正常完成训练 |

## 兼容性

- **向后兼容**：若 SFT 训练未触发早停（如高熵数据），`best_checkpoint.json` 不会生成 → DPO 侧自动回退到最终 checkpoint，行为与修复前完全一致
- **Checkpoint 兼容**：`best_checkpoint/` 下的 adapter 权重与最终 checkpoint 格式完全一致（均为 PeftModel adapter）
- **跨模型兼容**：回调不依赖特定模型架构，适用于任何 HF Trainer / TRL 训练器
