# Changelog 2026-05-01: Prompt Leakage Truncation Fix

## 修改文件
- evaluation/evaluator.py

## 变更内容
- 新增 _truncate_prompt_leakage(model_output)：在代码抽取前统一执行文本截断。
- 截断规则：
  - 若包含 ### Instruction:，则在该位置截断。
  - 若包含 ### Input:，则在该位置截断。
- 在评测主循环中，将模型原始输出 	ext 先清洗为 cleaned_for_extraction，再传入 extract_python_code_with_debug(...)。

## 变更目的
- 防止模型回显提示词模板（prompt leakage）污染代码抽取输入。
- 避免非代码提示段进入 extraction 导致 AST 解析失败，降低 invalid_extraction。

## 影响评估
- extraction 只接收清洗后的输出，行为更稳定且可复现。
- 该修复不修改 extraction 本身逻辑，仅在 evaluation 管线前置清洗。
- 每个样本都统一应用该规则，保证评测一致性。
