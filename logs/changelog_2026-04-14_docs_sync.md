# 2026-04-14 文档全量同步变更日志

## Added

- `logs/changelog_2026-04-14_docs_sync.md`
  - **Purpose**: 记录本次文档统一修订范围、原因与影响，满足每次更新必须产生日志的要求。
  - **Impact**: 后续可追踪文档与代码一致性的修复历史，降低维护歧义。

## Modified

- `README.md`
  - **What changed**: 补充标准树形结构（`project_root/`）、统一入口说明、执行流程（数据准备→训练→评测→汇总/可视化）。
  - **Purpose**: 让主文档直接对应当前代码结构与脚本职责。
  - **Impact**: 新成员可按单一路径快速理解项目并执行命令，减少误用旧脚本风险。

- `PROJECT_STRUCTURE.md`
  - **What changed**: 同步 `scripts/` 与 `configs/` 的当前文件清单，补充 `dpo_pairs.jsonl` 与准备脚本说明。
  - **Purpose**: 使目录说明与真实仓库保持一一对应。
  - **Impact**: 消除“文档有误/文件缺失”的结构冲突。

- `dataset/README.md`
  - **What changed**: 从旧版 `sql_security_dataset.json` 主入口说明，调整为当前主流程（`generate_expanded_dataset.py` + `data/combined/*`）。
  - **Purpose**: 对齐当前训练/评测的数据来源与配置字段。
  - **Impact**: 减少数据入口误解，避免按旧路径构建数据导致流程偏差。

- `scripts/README.md`
  - **What changed**: 更新为当前有效脚本清单，移除不存在目录的历史描述，统一入口术语。
  - **Purpose**: 让脚本职责与现存文件一致。
  - **Impact**: 使用者可准确找到配置准备、流水线、汇总和兼容评测脚本。

- `logs/README.md`
  - **What changed**: 示例命令改为 PowerShell 且补全 `--config` 参数。
  - **Purpose**: 与项目执行环境及主文档命令风格保持一致。
  - **Impact**: 复制即运行，降低日志采集误操作概率。

- `reports/operation_manual.md`
  - **What changed**: 替换不存在的旧脚本/旧配置命令，改为当前实际可运行流程与输出说明。
  - **Purpose**: 将历史手册修正为当前可执行手册。
  - **Impact**: 实验复现实操路径与代码一致，避免运行失败。

## Consistency Outcome

- 术语统一为：
  - 训练入口：`training/` 下各训练脚本
  - 评测入口：`evaluation/evaluate.py`
  - 汇总入口：`scripts/compare_results.py`
- 所有更新命令均采用 PowerShell 形式，且引用现存脚本与配置文件。
