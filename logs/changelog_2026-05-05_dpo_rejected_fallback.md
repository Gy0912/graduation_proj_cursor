# Changelog — 2026-05-05 DPO 被拒代码构造回退（十四次加固）

> 修复 `_vulnerable_variant_from_chosen()` 三种正则策略全部失败时 `raise ValueError`
> 导致整个 `build_dpo_pairs` / `main()` 脚本崩溃的问题。改为返回 `None` + 调用方
> 回退到 `_dispatch_vulnerable()` 从头生成脆弱代码。

---

## 1. 背景与危害（Problem）

### Bug #5：DPO 被拒代码构造依赖脆弱正则

**文件/模块**：
- `dataset/generate_expanded_dataset.py → _vulnerable_variant_from_chosen()`
- 辅助函数：`_try_variant_sqlalchemy()` / `_try_variant_sqlite_pymysql_percent()` / `_try_variant_indirect_full_query()`（约第 670-930 行）

**根因**：被拒代码通过正则将安全 chosen 代码转为脆弱形态。三种策略均依赖精确正则匹配：

| 策略                                  | 匹配模式                                                        |
| ------------------------------------- | --------------------------------------------------------------- |
| `_try_variant_sqlalchemy`             | `stmt = text("SELECT ... :v")` + `session.execute(stmt, {...})` |
| `_try_variant_sqlite_pymysql_percent` | `sql = "SELECT ... %s"` + `cur.execute(sql, (param,))`          |
| `_try_variant_indirect_full_query`    | `_full_query()` + `cur.execute(sql, (param,))`                  |

若 chosen 的安全代码与此三种模式有**轻微偏离**（不同缩进、不同变量名、不同的安全模式），则全部三种策略返回 `None`。旧版代码：

```python
# 旧版 _vulnerable_variant_from_chosen 末尾
raise ValueError(
    f"build_dpo_pairs: 无法从 chosen 生成脆弱变体（attack={attack!r}, "
    f"difficulty={_difficulty!r}）。"
)
```

**调用链**：
```
main() 
  → build_dpo_pairs(train, rng)          # 无 try/except 包裹
    → for r in train_rows:
        → _vulnerable_variant_from_chosen(...)  # 单行失败 → 整体崩溃
```

**严重性**：🟠 高（类别 A — 格式/解析）

**后果**：数据生成期间的静默崩溃。由于 `build_dpo_pairs` 在 `main()` 中调用且不被
`try/except` 包裹，正则不匹配将导致数据集构建脚本**完全失败**，不生成任何 DPO 对。

---

## 2. 修复方案（Solution）

### 2.1 `_vulnerable_variant_from_chosen()` — 返回 `None` 替代抛异常

```diff
-def _vulnerable_variant_from_chosen(...) -> str:
+def _vulnerable_variant_from_chosen(...) -> str | None:

-    raise ValueError(
-        f"build_dpo_pairs: 无法从 chosen 生成脆弱变体..."
-    )
+    # 三种正则策略均失败 → 返回 None，由调用方走 _dispatch_vulnerable 从头生成
+    return None
```

返回值语义：
- `str`：正则转换成功，返回脆弱代码
- `None`：三种策略均失败，调用方应使用 fallback

### 2.2 `build_dpo_pairs()` — 回退到 `_dispatch_vulnerable()`

当 `_vulnerable_variant_from_chosen()` 返回 `None` 时：

```python
rejected_raw = _vulnerable_variant_from_chosen(chosen_body, atk, diff, rng)
if rejected_raw is None:
    print(
        f"[DPO fallback] regex strategies exhausted for "
        f"attack={atk!r} difficulty={diff!r} "
        f"table={schema_table!r} col={schema_column!r} — "
        f"using _dispatch_vulnerable instead"
    )
    fallback_count += 1
    rejected_raw = _dispatch_vulnerable(atk, schema_table, schema_column, diff, rng)
```

`_dispatch_vulnerable()` 是从头生成的通用脆弱代码生成器，不依赖 chosen 代码结构：
- 接受 `attack` / `table` / `col` / `difficulty` 参数
- 覆盖全部 7 种攻击类型（`string_concat` / `fstring` / `format_string` / `fake_sanitization` / `orm_misuse` / `indirect_injection` / `parameterized_query`）
- 产出保证通过 `ast.parse` 且命中脆弱 SQL 模式的代码

**末尾 summary 日志**：

```
[DPO] fallback summary: 23/800 rows (2.88%) used _dispatch_vulnerable
     (regex transformation failed for chosen code)
```

### 2.3 为什么不放宽正则

放宽正则（如 `.` 匹配任意变量名）会让脆弱代码构造产生语法错误或语义错误。与其
维护脆弱正则在"兼容更多 chosen 形态"与"保证产出正确脆弱代码"之间的张力，不如
在正则失效时直接走已验证的 `_dispatch_vulnerable` 路径——维护成本更低、正确性保证更强。

---

## 3. 影响范围（Impact）

| 变更文件                               | 变更类型                                                         | 下游影响                                       |
| -------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `dataset/generate_expanded_dataset.py` | `_vulnerable_variant_from_chosen()` 返回类型 `str` → `str\|None` | 调用方需处理 `None`                            |
| `dataset/generate_expanded_dataset.py` | `build_dpo_pairs()` 新增 fallback 路径                           | DPO 对数量不变，个别 rejected 可能来自通用模板 |
| `dataset/generate_expanded_dataset.py` | `build_dpo_pairs()` 新增 `fallback_count` 计数器 + summary 日志  | 仅日志输出，不影响数据                         |

**不触碰的部分**：
- 三种 `_try_variant_*` 正则策略内部逻辑零改动
- `_dispatch_vulnerable()` 零改动
- DPO 训练管线零改动
- 评测管线零改动

---

## 4. 使用指南

**正常情况**：绝大多数 chosen 代码匹配至少一种正则策略 → fallback 计数为 0。

**fallback 触发时**（如新增安全代码模板、修改 chosen 生成逻辑后）：
- 检查 `[DPO fallback]` 日志确认哪些 `(attack, difficulty, table, col)` 触发
- 若 fallback 比例 > 10-20%，考虑为新的 chosen 模式新增一条 `_try_variant_*` 策略
- fallback 产出的 DPO 对仍然有效（`_dispatch_vulnerable` 保证脆弱且语法正确）

---

## 5. 测试验证（Verification）

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 2.20s ==============
```

---

## 6. 相关文档

- `README.md` — 「关键修复（十四）」条目
- `dataset/generate_expanded_dataset.py` — `_vulnerable_variant_from_chosen()` / `build_dpo_pairs()` / `_dispatch_vulnerable()`
