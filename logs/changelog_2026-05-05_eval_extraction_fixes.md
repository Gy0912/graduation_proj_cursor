# Changelog — 2026-05-05 评测代码抽取三项修复（十一次加固）

> 修复评测管线中三项格式/解析类别 Bug，消除提示泄漏检测失效、
> 代码抽取不一致与缺失去重回退三个根因，使评测端行为与数据集端对齐。

---

## 1. 背景与危害（Problem）

本次修复解决三个独立但可复合的评测管线缺陷：

| 问题                              | 文件/函数                                                                                                        | 类别           | 严重性 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------- | ------ |
| #1 提示泄漏检测因格式不匹配失效   | `evaluator.py:_truncate_prompt_leakage()` / `sql_injection_detector.py:extract_python_code_with_debug()`         | A（格式/解析） | 🔴 严重 |
| #2 数据集端与评测端代码抽取不一致 | `adversarial.py:_first_fenced_python_or_whole()` vs `sql_injection_detector.py:extract_python_code_with_debug()` | A（格式/解析） | 🔴 严重 |
| #9 评测端缺失去重回退             | `adversarial.py:_collapse_identical_halves()` vs `sql_injection_detector.py:extract_python_code_with_debug()`    | A（格式/解析） | 🟡 中   |

三者复合危害：模型输出提示接续（Bug #1）→ 产生无围栏纯 Python 文本 → 评测端无回退机制（Bug #2）→ extraction_failure_rate 接近 1.0 → `aggregate_metrics` 抛出 `RuntimeError`（阈值 0.5），整个评测运行中止。

---

## 2. 根因分析（Root Cause）

### 2.1 Bug #1：提示泄漏检测格式不匹配

**两处搜索目标**：`"### Instruction:"` 与 `"### Input:"`。

**实际提示模板**（训练/评测/DPO 全线使用）：
```python
# sft_preprocess.py:row_to_prompt_completion
template = "Instruction:\n{instruction}\n\nInput:\n{input}\n\n"

# generate_expanded_dataset.py:training_prompt()
"Instruction:\n" + instruction + "\n\nInput:\n" + input + "\n\n"
```

`### Instruction` 变体**仅存于死代码** `common.py:sft_format` 中——全流水线无任何调用点。

**后果**：若模型重复或接续提示模式（LLM 常见行为），`"Instruction:\nWrite Python database..."` 等文本会原样进入代码抽取环节，两个 `find("### Instruction:")` 永远返回 `-1`，泄漏检测形同虚设。

### 2.2 Bug #2：代码抽取回退机制不一致

**数据集端** `adversarial.py:_first_fenced_python_or_whole()`：
- 有 fenced block → 返回 fence 内容
- 无 fenced block → `ast.parse(全文)`，成功则返回全文

**评测端** `sql_injection_detector.py:extract_python_code_with_debug()`：
- 有 fenced block → 取最后一个可 parse 的
- 无 fenced block → **立即**返回 `(None, None, "no code found", "python_fence")`

**后果**：若模型输出纯 Python 代码（无围栏——这正是 SFT 训练目标的格式），每条样本均抽取失败 → `extraction_failure_rate` 接近 1.0 → 评测中止。这是**保证会导致故障**的路径。

### 2.3 Bug #9：缺失去重回退

**数据集端** `adversarial.py:_collapse_identical_halves()`：
- 将重复的 Python 代码折叠为单份（如 `code_code_` → `code_`）
- 在 `extract_code_only_completion` 中被调用

**评测端** `sql_injection_detector.py:extract_python_code_with_debug()`：
- 无此逻辑

**后果**：若模型退化并重复输出，数据集管道的去重会掩盖问题（训练准备中不可见），但评测管道将其视为原始文本，导致 `ast.parse` 失败。

---

## 3. 修复方案（Solution）

### 3.1 `evaluation/evaluator.py` — `_truncate_prompt_leakage()`

```diff
- instruction_idx = text.find("### Instruction:")
+ instruction_idx = text.find("Instruction:\n")
- input_idx = text.find("### Input:")
+ input_idx = text.find("\nInput:\n")
```

### 3.2 `detection/sql_injection_detector.py` — 三项修复合一

**新增本地 `_collapse_identical_halves()` 副本**：与 `dataset/adversarial.py` 版本平行维护，评测管线不能依赖数据集模块。

**`extract_python_code_with_debug()` 修改**：

1. **提示泄漏截断**（Fix #1）：同上，改用 `"Instruction:\n"` / `"\nInput:\n"` 匹配。
2. **全文回退**（Fix #2）：无围栏代码块时，尝试 `ast.parse` 全文（source=`"full_text_fallback"`）。
3. **去重折叠**（Fix #9）：fenced 候选码与全文回退文本在 `ast.parse` 前均经 `_collapse_identical_halves` 处理。

```python
# 无 fenced block 时的回退路径
if not code_blocks:
    clean = text.strip()
    if clean:
        deduped = _collapse_identical_halves(clean)
        ok, reason = _parse_python(deduped)
        if ok:
            return ExtractionResult(deduped, deduped, "ok", "full_text_fallback")
        return ExtractionResult(None, deduped, reason, "full_text_fallback")
    return ExtractionResult(None, None, "no code found", "python_fence")

# fenced block 候选也经去重
for candidate in reversed(code_blocks):
    deduped = _collapse_identical_halves(candidate)
    ...
```

---

## 4. 影响范围（Impact）

| 变更文件                              | 变更类型                                      | 下游影响                                              |
| ------------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| `evaluation/evaluator.py`             | `_truncate_prompt_leakage()` 搜索字面量       | 提示泄漏截断现在正确匹配实际模板                      |
| `detection/sql_injection_detector.py` | 新增 `_collapse_identical_halves()`           | 评测端获得与数据集端一致的去重能力                    |
| `detection/sql_injection_detector.py` | `extract_python_code_with_debug()` 三合一修复 | 无围栏代码现在可被正确抽取；重复输出不再导致 AST 失败 |

**不触碰的部分**：
- `detection/sql_injection_detector.py::extract_python_code()`（死代码，无调用者）未改动
- `dataset/adversarial.py` 无任何变更
- 检测逻辑（Bandit / 规则 / 污点）零改动
- 指标聚合逻辑零改动

---

## 5. 测试验证（Verification）

全部 61 条既有测试通过，无回归：

```
============================= test session starts =============================
collected 61 items
...
============== 61 passed, 6 warnings, 4 subtests passed in 2.43s ==============
```

---

## 6. 相关文档

- `README.md` — 「关键修复（十一）」条目
- `dataset/adversarial.py` — `_first_fenced_python_or_whole()` / `_collapse_identical_halves()` 参考实现
- `training/sft_preprocess.py` — `row_to_prompt_completion()` 实际提示模板
- `dataset/generate_expanded_dataset.py` — `training_prompt()` 评测/DPO 提示模板
- `training/common.py` — `sft_format()`（死代码，`### Instruction` 变体唯一出处）
