## 变更标题
DPO 数据预处理 `dataset.map` 多进程导致 accelerate state 未初始化报错修复

## 修改内容
- **修改** `training/stable_dpo_trainer.py`
  - 在 `StableDPOTrainer.__init__` 中新增 `dataset_num_proc` 兜底逻辑：
    - 当外部配置或调用传入 `dataset_num_proc > 1` 时，自动覆盖为 `1`。
  - 保留告警输出（通过 `self.accelerator.print`），不全局移除日志能力。

## 变更目的
- 解决 TRL `DPOTrainer` 在 `dataset.map(tokenize_fn)` 多进程场景下，
  子进程触发 `logger.warning` 时因 accelerate `PartialState` 未初始化导致的运行时错误。

## 影响评估
- **DPO 训练**：仅影响数据预处理并行度，不改变模型结构、损失计算或优化逻辑，训练正确性保持不变。
- **SFT 训练**：未修改 `training/train_lora_sft.py`、`training/train_qlora_sft.py` 或 `training/sft_preprocess.py`，无行为变化。
- **评估流程**：未修改 `evaluation/` 相关代码与调用路径，评估逻辑不变。

## 兼容性说明
- 与 TRL `DPOTrainer` 保持兼容：修复位于自定义 `StableDPOTrainer`，不改动第三方包源码，不影响后续依赖升级与环境复现。
