## 变更标题

DPO 数据集构建：`chosen` / `rejected` 规范为纯 Python 并经 `ast.parse` 校验

## 修改内容

- **修改** `dataset/adversarial.py`
  - 新增 `extract_code_only_completion`（及内部 `_extract_safe_solution_section_text`、`_collapse_identical_halves`、`_first_fenced_python_or_whole`）：从三段式或裸代码文本得到无 marker、无 fence 的 Python 字符串；作为 SFT/DPO 的**单一实现源**。
- **修改** `dataset/generate_expanded_dataset.py`
  - `build_dpo_pairs`：`chosen` 由 `extract_code_only_completion(str(output))` 得到；`rejected` 由 `extract_code_only_completion(_dispatch_vulnerable(...))` 或回退为 strip 后的合成串；二者均 **`ast.parse`** 失败时 `ValueError`（生成器 FAIL FAST）。
  - 模块文档字符串与 `import ast` 同步更新。
- **修改** `training/sft_preprocess.py`
  - 删除重复的 fence/抽取实现，改为 `from dataset.adversarial import extract_code_only_completion`；`normalize_sft_records_for_training` 行为不变。

## 变更目的

- 使 DPO 偏好对与 SFT、评测侧**代码抽取语义**一致：两臂均为可解析的纯 Python，避免 TRL 在 token 化长段自然语言与 marker 上浪费容量或引入分布噪声。
- **不改变** `prompt` 字段与任何 `training/*dpo*.py` 训练循环逻辑。

## 影响评估

- 需**重新运行** `dataset/generate_expanded_dataset.py` 以重写 `data/dpo_pairs.json`；旧文件若仍含三段式 `chosen`，与当前训练预期不一致。
- **未修改** `evaluation/` 与模型结构代码。
