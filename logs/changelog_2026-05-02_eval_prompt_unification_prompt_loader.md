# Changelog: Evaluation prompt unification（`prompt_loader`）（2026-05-02）

## 修改

- **`evaluation/prompt_loader.py`**
  - **删除** `_instruction_input_prompt` 及「Generate one Python module…」等旧英文模板。
  - **新增** `build_eval_code_only_prompt(instruction, user_input)`：唯一格式为  
    `Instruction:\n…\n\nInput:\n…\n\nOutput ONLY valid Python code.\n`，与 `dataset/generate_expanded_dataset.py::training_prompt` 的前缀一致，并追加单行 code-only 约束（无 `###`、无 Output contract、无分段式输出要求）。
  - **`_normalize_sample`**：只要 `instruction` 与/或 `input`/`input_code` 非空，**一律**用 `build_eval_code_only_prompt` 重写 `prompt`，**忽略** JSON 中旧版 `prompt` 字段，避免磁盘上的 `### Instruction` / Output contract 进入评测管线。
  - **回退**：仅当 instruction 与 input 系列全空且存在非空顶层 `prompt` 时，仍使用该 `prompt`（兼容极少数仅 prompt 的遗留文件）。

## 目的与影响

- **目的**：评测侧 prompt 与 SFT/DPO 的 `Instruction`/`Input` 风格对齐，并固定 code-only 尾句；与 `generation.code_only_inference=True` 时 `append_code_only_instruction` 的子串去重配合，避免重复追加长后缀。
- **影响**：`load_eval_prompts` 产出的每条样本 `prompt` 形态统一；无需重生成 `eval_expanded.json` 即可在加载阶段覆盖旧 `prompt` 文本。
- **未改**：`evaluation/inference_constraints.py`（`append_code_only_instruction` 逻辑保持；读确认子串匹配仍生效）。

## 手工校验示例

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0,'.'); from evaluation.prompt_loader import load_eval_prompts; s=load_eval_prompts(Path('data/eval_expanded.json')); print(s[0]['prompt'])"
```

预期：`###` / `Output contract` / `Generate one Python` 均不应出现；尾行为 `Output ONLY valid Python code.`。
