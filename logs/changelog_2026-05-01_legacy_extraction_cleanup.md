# Changelog 2026-05-01: Legacy Extraction Cleanup

## 修改文件
- detection/sql_injection_detector.py

## 删除内容
- 删除 SAFE SOLUTION 分段提取路径及相关辅助逻辑。
- 删除 full-text AST 回退提取路径。
- 删除与旧提取分支绑定的辅助函数：
  - _extract_safe_solution_section
  - _extract_from_safe_solution
  - _extract_python_fences
  - _first_valid_candidate

## 保留并统一的提取逻辑
- 仅从  `python ... `  fenced block 提取候选代码。
- 按从后到前顺序对候选执行 st.parse 校验。
- 首个可解析候选立即返回。
- 若无代码块或全部校验失败，返回 None。

## 目的与影响
- 目的：彻底移除历史启发式/回退式提取路径，避免部分样本走不同分支导致行为不一致。
- 影响：提取行为简化为单一路径，结果可预测、可审计，不再误把说明文本当代码。
