# Changelog 2026-05-01: Evaluation Prompt Python-Only

## 修改文件
- `evaluation/prompt_loader.py`

## 修改内容
- 仅修改评测 prompt 构造函数 `_instruction_input_prompt(instruction, user_input)`。
- 删除三段式输出要求与相关 marker 指令：`[SECURITY WARNING]`、`[EXPLANATION]`、`[SAFE SOLUTION]`。
- 删除 markdown/代码围栏导向和 `### Instruction` / `### Input` / `### Response` 结构化提示。
- 新 prompt 明确要求模型输出：
  - 仅 Python 代码
  - 纯文本（无 markdown / 无代码块）
  - 无解释文本
  - 无分节标题
  - 整体可被 `ast.parse` 解析

## 目的与影响
- 目的：从评测输入源头约束模型输出形态，减少 prompt 泄漏与结构化噪声进入抽取链路。
- 影响：评测阶段生成更接近“纯 Python 代码”目标，降低无关文本污染提取导致的失败概率。
- 范围：仅评测 prompt 模板变更；未修改 extractor 或 evaluation 统计逻辑。
