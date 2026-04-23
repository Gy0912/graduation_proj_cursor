# 2026-04-14 代码清理与入口统一变更日志

## Added

- `logs/changelog_2026-04-14_entrypoint_cleanup.md`
  - **Purpose**: 记录本次最小化重构的具体内容、原因与影响，满足“每次更新必须有变更日志”的要求。
  - **Impact**: 后续审计可直接追踪本次删减是否触及训练/评测核心逻辑，降低维护沟通成本。

## Modified

- `README.md`
  - **What changed**: 新增“统一入口（已精简）”与“手动验收（行为保持不变）”两节，明确唯一训练/评测入口与 PowerShell 验证命令。
  - **Purpose**: 降低入口混淆，避免继续调用已移除的兼容脚本。
  - **Impact**: 用户可按单一入口执行训练与评测，命令路径更一致，结果文件格式不变。

- `PROJECT_STRUCTURE.md`
  - **What changed**: 从目录树与文件职责中移除已删除的兼容脚本条目。
  - **Purpose**: 保持结构文档与实际仓库一致，避免误导。
  - **Impact**: 文档准确性提升，不再出现“文档有、仓库无”的入口说明冲突。

- `scripts/README.md`
  - **What changed**: 删除 `run_baseline.py` 的说明项。
  - **Purpose**: 与入口统一策略保持一致，避免保留历史包装入口描述。
  - **Impact**: scripts 说明更聚焦真实在用脚本。

## Removed

- `training/train_dpo.py`
  - **Reason**: 仅转发到 `training/dpo_train.py`，不包含独立训练逻辑。
  - **Impact**: 消除重复 DPO 入口，降低命令歧义；DPO 实际训练行为不变。

- `training/train_lora_dpo.py`
  - **Reason**: 同样仅转发到 `training/dpo_train.py`，与主入口重复。
  - **Impact**: 训练入口收敛为单一文件，后续维护仅需修改 `dpo_train.py`。

- `scripts/run_baseline.py`
  - **Reason**: 仅转发到 `evaluation/evaluate.py --model baseline`，属于冗余兼容层。
  - **Impact**: baseline 评测统一走 `evaluation/evaluate.py`，评测输出与指标计算逻辑保持不变。

## Functional Safety Notes

- 本次未改动模型构建、训练参数读取、DPO/QLoRA 训练流程、评测检测逻辑与输出 JSON 结构。
- 变更集中在“入口文件去重 + 文档同步”，属于低风险清理。
