# Changelog — 2026-05-05 EOS 截断防止多代码块（十八次加固）

> 在生成的 token 序列中定位第一个 EOS token 并在解码前截断，防止模型在有效代码后
> 继续生成第二个（可能不完整或格式错误的）代码块进入代码抽取器。

---

## 1. 背景与危害（Problem）

### Bug #10：无 EOS 后截断——模型可能生成多个独立代码块

**文件**：
- `evaluation/evaluator.py → run_eval_on_prompts()`（第 542-543 行）
- `detection/sql_injection_detector.py → extract_python_code_with_debug()`

**根因**：

旧版解码流程：
```python
gen_tokens = out_ids[row, prompt_len:]
text = tok.decode(gen_tokens, skip_special_tokens=True)
```

`skip_special_tokens=True` 会跳过 EOS token。若模型在输出有效代码后**未生成 EOS**
（因训练或 `max_new_tokens` 限制），可能会在 token 序列末尾开始生成第二个代码块。

此时 `extract_python_code_with_debug` 的 `reversed(code_blocks)` 逻辑会取**最后一个**
围栏代码块——可能是第二个不完整/格式错误的块。

**示例**：
```
import sqlite3\ndef fetch(): ...\n```\n\n```python\n# incomplete second block
```

提取器取最后一个 ` ```python ` 块 → 不完整代码 → AST 解析失败 → 抽取失败。

**严重性**：🟡 中（类别 D — 解码/推理）

---

## 2. 修复方案（Solution）

在解码前定位第一个 EOS token 并截断：

```python
gen_tokens_full = out_ids[row, prompt_len:]
gen_list = gen_tokens_full.tolist()
eos_pos = None
if tok.eos_token_id is not None:
    try:
        eos_pos = gen_list.index(tok.eos_token_id)
    except ValueError:
        pass
gen_tokens = gen_tokens_full[:eos_pos] if (eos_pos is not None and eos_pos > 0) else gen_tokens_full
text = tok.decode(gen_tokens, skip_special_tokens=True)
```

**三种情况的处理**：

| 情况                                    | EOS 位置                    | 行为                                      |
| --------------------------------------- | --------------------------- | ----------------------------------------- |
| 正常：有效代码 + EOS                    | pos > 0 且 token 列表中找到 | 截断到 EOS 前 → 仅解码有效代码            |
| 超长：有效代码 + 第二个块起始（无 EOS） | `eos_pos is None`           | 不截断 → 全量解码（抽取器回退逻辑处理）   |
| 退化：空输出                            | pos == 0                    | 不截断 → 空文本 → extraction returns None |

---

## 3. 影响范围（Impact）

| 文件                      | 变更                                        | 影响                           |
| ------------------------- | ------------------------------------------- | ------------------------------ |
| `evaluation/evaluator.py` | `run_eval_on_prompts()` 解码前新增 EOS 截断 | 仅评测端，训练/检测/DPO 零改动 |

**不触碰**：
- `extract_python_code_with_debug()` 零改动（截断在上游完成）
- `model.generate()` 零改动
- 训练管线零改动

---

## 4. 测试验证（Verification）

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.15s ==============
```

---

## 5. 相关文档

- `README.md` — 「关键修复（十八）」条目
- `evaluation/evaluator.py` — `run_eval_on_prompts()` 解码段
