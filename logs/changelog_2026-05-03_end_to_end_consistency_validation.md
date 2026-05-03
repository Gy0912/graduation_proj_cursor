## 变更标题

End-to-end consistency validation（Dataset / SFT / DPO / Eval）与 README 流水线定稿。

## 变更内容

### 1) 修改 `README.md`

- 新增 `Final pipeline architecture` 小节，明确了四阶段职责与边界：
  - Dataset：产出 `train/eval/dpo` 三类数据，`output` 为可执行 Python；
  - SFT：训练前 code-only 规范化与语法门闸；
  - DPO：同 prompt/schema 的 aligned pairs（安全 chosen vs 脆弱 rejected）；
  - Eval：统一入口、强校验、检测与指标聚合。
- 新增 `Manual steps (PowerShell)`，给出完整命令链：
  - 生成数据集；
  - 训练 SFT；
  - 训练 DPO；
  - 运行 Eval。

### 2) 一致性验证（本次审计结论）

- Dataset：当前 `data/train_expanded.json` 抽样与全量脚本检查显示 `output` 为可 `ast.parse` 的 Python，未出现解释性 marker 文本。
- SFT：流程中仍保留 code-only 抽取逻辑（说明“无需抽取”条件未满足）；但基于当前训练集进行等价检查，格式导致的潜在丢样数为 0。
- DPO：`data/dpo_pairs.json` 经项目内 SQLi 模式检测器验证，`chosen` 不命中脆弱模式、`rejected` 命中脆弱模式，pair 方向一致。
- Eval：`evaluation/prompt_loader.py` 会统一重建 `prompt` 并附加 `Output ONLY valid Python code.`，与训练 `training_prompt`（不含该尾句）存在模板差异；同时该尾句属于结构化指令风格约束。

## 目的与影响

- 目的：将“端到端一致性”从口头要求落到可追溯文档与可复现实操命令，避免训练/评测口径漂移。
- 影响：
  - 正向：README 现在可直接作为 pipeline 运行与审计入口；
  - 风险暴露：SFT 与 Eval 仍存在“抽取依赖”和“prompt 模板尾句差异”两处未完全一致项，后续可据此做 targeted 修复。

## 清理说明

- 本次更新未新增临时产物、缓存或冗余副本文件；
- 主目录未引入旧版本文件，保持当前单一有效版本结构。
