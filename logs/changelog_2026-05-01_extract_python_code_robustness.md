# Changelog 2026-05-01: extract_python_code Robustness

## 修改文件
- `detection/sql_injection_detector.py`

## 修改内容
- 仅修改 `extract_python_code(model_output: str) -> str | None`。
- 新增 marker 前缀裁剪：当文本中出现 `### Response` / `### Instruction` / `### Input` 时，从最后出现的 marker 开始保留文本，忽略其前内容。
- 提取策略改为只取最后一个 ` ```python ... ``` ` 代码块，不再遍历多个块并逐个回退。
- 对提取出的最后代码块执行 `ast.parse` 校验，语法失败时返回 `None`。

## 目的与影响
- 目的：避免提示模板、解释文本或多段混合输出干扰代码抽取，减少 invalid extraction。
- 影响：提取行为更确定（deterministic），只接受最终代码候选，且必须是可解析 Python。
