# Changelog — 2026-05-05 回退提示格式对齐（二十次加固）

> 将 `prompt_loader.py::_instruction_input_prompt()` 的回退提示格式从
> 自定义 "Generate one Python module..." / "no fences" 指令改为与训练端
> `training_prompt()` 一致的标准格式。消除评测分布偏移的潜在风险。

---

## 1. 背景与危害（Problem）

### Bug #12：回退提示包含与训练数据冲突的「no fences」指令

**文件**：`evaluation/prompt_loader.py → _instruction_input_prompt()`

**旧版回退提示**：

```python
"Generate one Python module that satisfies the request below.\n"
"Return only Python code as plain text.\n"
"Do not output markdown, code fences, explanations, or section headers.\n"
"The entire output must be valid Python syntax parsable by ast.parse.\n"
f"Request: {(instruction or '').strip()}\n"
f"Context: {(user_input or '').strip()}\n"
```

**训练端提示格式**（`training_prompt()`）：

```python
"Instruction:\n" + instruction.strip() + "\n\nInput:\n" + input.strip() + "\n\n"
```

**三项不一致**：

| 维度 | 回退格式                                                  | 训练格式                  | 问题                            |
| ---- | --------------------------------------------------------- | ------------------------- | ------------------------------- |
| 结构 | "Generate one Python module..." + "Request:" + "Context:" | "Instruction:" + "Input:" | 完全不同的提示模板              |
| 指令 | "Do not output markdown, code fences..."                  | 无此类指令                | 模型从未被训练遵循「no fences」 |
| 风格 | 指令式 prose                                              | 简洁的 key:value 对       | 分布不一致                      |

**当前状态**：由于 `to_eval_prompt_row` 始终写入 `prompt` 字段，此回退尚未被触发。
但它是一颗定时炸弹——若评测数据集因任何原因缺失 `prompt` 字段，提示分布将剧烈偏移。

**严重性**：🟢 低（当前被 prompt 字段存在性掩盖；类别 C — 分布偏移潜在风险）

---

## 2. 修复方案（Solution）

将回退提示改为与训练端完全一致的格式：

```python
def _instruction_input_prompt(instruction: str, user_input: str) -> str:
    return (
        "Instruction:\n"
        + (instruction or "").strip()
        + "\n\nInput:\n"
        + (user_input or "").strip()
        + "\n\n"
    )
```

**与 `dataset/generate_expanded_dataset.py::training_prompt()` 格式一致**：
```python
def training_prompt(instruction: str, input_text: str) -> str:
    return (
        "Instruction:\n"
        + instruction.strip()
        + "\n\nInput:\n"
        + (input_text or "").strip()
        + "\n\n"
    )
```

---

## 3. 影响范围（Impact）

| 文件                          | 变更                                   | 影响                   |
| ----------------------------- | -------------------------------------- | ---------------------- |
| `evaluation/prompt_loader.py` | `_instruction_input_prompt()` 格式替换 | 回退路径与训练分布对齐 |

**不触碰**：
- `_normalize_sample()` 逻辑零改动（回退触发条件不变）
- `training_prompt()` 零改动
- 评测/训练/检测管线零改动

**当前无实际影响**：因 `prompt` 字段始终存在，此回退从未被触发。修复是预防性的。

---

## 4. 测试验证

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.21s ==============
```

---

## 5. 相关文档

- `README.md` — 「关键修复（二十）」条目
- `evaluation/prompt_loader.py` — `_instruction_input_prompt()` / `_normalize_sample()`
- `dataset/generate_expanded_dataset.py` — `training_prompt()`
