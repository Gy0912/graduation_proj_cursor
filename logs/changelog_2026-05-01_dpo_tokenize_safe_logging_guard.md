## 变更标题
修复 DPO 预处理阶段 accelerate logger 未初始化崩溃（跨平台安全日志护栏）

## 修改内容
- **修改** `training/stable_dpo_trainer.py`
  - 将 `dataset_num_proc=1` 的强制逻辑前移到 `super().__init__()` 之前，确保在 TRL 初始化触发 `dataset.map` 前生效。
  - 新增 `_safe_tokenize_warning_context()`：
    - 在 `DPOTrainer` 初始化期间，临时替换 TRL `dpo_trainer.logger.warning`。
    - 默认改为标准库 `logging` 输出，避免依赖 accelerate state。
    - 支持环境变量 `DPO_TOKENIZE_SILENT=1` 静默该类 warning（可选）。
  - 保留其余训练逻辑（DPO loss、模型结构、优化流程）不变。

## 变更目的
- 解决 `dataset.map(tokenize_fn)` 中 `logger.warning` 调用 accelerate logger 时，
  因执行上下文缺失 accelerate 初始化而报错的问题。
- 该问题在 Windows `spawn` 下更容易出现；本修复对 Linux / 多 GPU 也安全。

## 影响评估
- **DPO 训练**：仅调整预处理阶段日志通道与并行策略，不改变训练语义。
- **SFT 训练**：未修改 SFT 相关脚本与预处理流程，不受影响。
- **评估流程**：未修改 `evaluation/` 代码，不受影响。

## 结论
- `dataset_num_proc=1` 仍是必要条件（降低 map 并发复杂度），但单独使用并不总是充分；
  还需要在 tokenize 期间避免 accelerate logging 依赖，才能彻底消除该崩溃。
