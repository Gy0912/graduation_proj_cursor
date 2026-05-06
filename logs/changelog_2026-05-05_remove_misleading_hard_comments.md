# Changelog — 2026-05-05 移除硬样本误导性注释（十五次加固）

> 移除 `_decorate_hard_output()` 注入 SFT 训练目标的误导性安全注释与函数名。
> 这些表面装饰导致模型学会在安全代码旁生成暗示不安全的文本。

---

## 1. 背景与危害（Problem）

### Bug #6：`_decorate_hard_output` 向 SFT 训练目标注入误导性安全注释

**文件**：`dataset/generate_expanded_dataset.py → _decorate_hard_output()`

**旧代码**：

```python
def _decorate_hard_output(difficulty: str, code: str, rng: random.Random) -> str:
    if difficulty != "hard":
        return code
    prefix = rng.choice([
        "# sanitize_input() validated upstream — SQL fragment is trusted\n",
        "# reviewed: uses safe_query() wrapper (wrapper still builds SQL)\n",
        "# ORM migration pending; keep legacy string assembly\n",
    ])
    if "def " in code and rng.random() < 0.4:
        code = code.replace("def fetch_rows", "def safe_query", 1)
    return prefix + code
```

**问题**：
1. 误导性注释（如 `# ORM migration pending; keep legacy string assembly`）与安全代码一同进入 SFT code-only 训练目标
2. 函数重命名（`def fetch_rows` → `def safe_query`）将不安全暗示嵌入函数命名
3. 模型学会在安全代码旁生成暗示不安全的注释 → 评测阶段可能触发假阳性

**后果**：
- 规则检测器或 Bandit 可能将 `safe_query()` 函数名视为可疑
- 安全代码假阳性增加（FP 上升）
- 注释使 `ast.parse` 有效但混淆了无 `sql_injection` 模式的代码的语义含义

**严重性**：🟡 中（类别 B — 训练目标/标签）

---

## 2. 修复方案（Solution）

移除所有误导性表面装饰。Hard 样本的难度差异完全由 `_hard_safe_reference()` 的代码结构
（多函数、间接调用等）体现——这才是真正的难度来源，无需额外的注释/命名层面误导。

```python
def _decorate_hard_output(difficulty: str, code: str, rng: random.Random) -> str:
    """hard：增强代码复杂度（多函数、间接调用等）。

    2026-05-05 修复（问题 #6）：旧版在此注入误导性注释与误导性函数名，
    现已移除所有误导性装饰——hard 样本的难度差异完全由 _hard_safe_reference
    的代码结构体现。
    """
    _ = (difficulty, rng)
    return code
```

---

## 3. 影响范围（Impact）

| 文件                                   | 变更                                 | 影响                           |
| -------------------------------------- | ------------------------------------ | ------------------------------ |
| `dataset/generate_expanded_dataset.py` | `_decorate_hard_output()` 改为无操作 | Hard 样本不再含误导性注释/命名 |

**不触碰**：
- `_hard_safe_reference()` 零改动（代码结构难度保留）
- `_make_safe_sft_output()` 零改动（仍调用 `_decorate_hard_output`，但函数变成无操作）
- 评测/训练/检测管线零改动

---

## 4. 测试验证

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.21s ==============
```

---

## 5. 相关文档

- `README.md` — 「关键修复（十五）」条目
- `dataset/generate_expanded_dataset.py` — `_decorate_hard_output()` / `_hard_safe_reference()`
