# Changelog：`CODEBASE_DETAILED_OVERVIEW.md` 扩写为论文级（2026-05-10）

## 修改

- **`CODEBASE_DETAILED_OVERVIEW.md`**
  - **内容**：在原「工程全景」基础上扩写为学位论文可用的方法/系统叙述，新增：**摘要**；**威胁模型与假设**；**符号表**；**Mermaid 端到端数据流**；数据集 schema/分布/`eval_fixed` 单一写入者的**形式化说明**；模板库 v2 与重要性采样的**方法论描述**；对抗与 **code-only** 监督的**教学设计论证**；**DPO 目标、同构偏好、tokenizer 契约**；**LoRA/QLoRA/SFT 损失**的概念公式；**推理与解码**（重复惩罚、EOS 截断、`code_only_training`）；检测子系统 **merge_mode**；**代码抽取优先级**；评测指标 **valid/conservative/strict、主指标对、invalid 硬阈值、响应质量**等形式化与层级说明；**实验因素表**；**局限性与参考文献占位**；附录（配置键、产物路径）。
  - **目的**：满足「论文级别的详细」叙述需求，读者可在不翻阅分散 README 补丁的情况下撰写方法章节草稿。
  - **影响**：仅为文档加厚；不涉及训练、评测或可执行脚本行为变更。
