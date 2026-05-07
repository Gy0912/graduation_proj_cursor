# P0-1 / P1-5 修复：chosen/rejected DPO 语义同构崩塌 + 回退路径语义断裂

**日期**：2026-05-06  
**严重级别**：P0（DPO 训练信号退化）+ P1（回退路径独立生成）  
**影响范围**：`dataset/generate_expanded_dataset.py` — `_vulnerable_variant_from_chosen()` 与被调函数

## 覆盖的 Bug

| Bug ID   | 描述                                                                                | 本文覆盖     |
| -------- | ----------------------------------------------------------------------------------- | ------------ |
| **P0-1** | chosen/rejected 语义同构崩塌——正则失败后回退生成完全不同结构的脆弱代码              | § 优先级 1-2 |
| **P1-5** | `_dispatch_vulnerable` 回退路径语义断裂——生成独立代码结构（不同函数名/签名/import） | § 优先级 3   |

## 问题

DPO 训练对中的 `chosen`（安全）与 `rejected`（脆弱）代码存在**语义同构崩塌**：当脆弱化变换失败时，回退路径生成完全独立的脆弱代码（不同 import、不同函数签名、不同控制流），导致 DPO 偏好信号从「同一任务：安全实现 > 脆弱变体」退化为「任意安全代码 > 任意脆弱代码」。

### 根本原因

三种正则策略（`_try_variant_sqlalchemy`、`_try_variant_sqlite_pymysql_percent`、`_try_variant_indirect_full_query`）使用极其精确的代码模板匹配：

```python
# _try_variant_sqlalchemy 的核心正则
stmt_m = re.search(
    r'stmt\s*=\s*text\("(SELECT \* FROM \w+ WHERE \w+ = ):(\w+)"\)\s*\n'
    r'\s*return\s+session\.execute\(\s*stmt\s*,\s*(\{[^}]+\})\s*\)\.fetchall\(\)',
    code, re.DOTALL,
)
```

该正则要求 `stmt = text(...)` 后**恰好紧跟** `return session.execute(stmt, {...}).fetchall()`，且中间不能有空行、注释或额外语句。任何缩进变化、变量名差异、中间插入语句都会导致匹配失败。

当三种策略全部失败时，回退到 `_dispatch_vulnerable()`——该函数生成**完全独立**的脆弱代码，import、函数签名、控制流与 chosen 完全不同。

### 实际影响

从 `dpo_pairs.json` 中可观察到的证据：

| chosen（安全）                                  | rejected（脆弱，回退路径）                                   |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `import pymysql` + `cur.execute(sql, (value,))` | `from sqlalchemy import text` + `session.execute(text(...))` |
| 参数化查询（pymysql %s 占位符）                 | 字符串拼接（SQLAlchemy text()）                              |

→ import 范式完全不同的 DPO 对。

## 修复策略

**核心原则**：chosen 与 rejected 应共享相同的 import、函数签名和控制流——唯一区别是 SQL 构造方式（参数化 vs 拼接）。

### 优先级 1：AST 手术变换（新增）

新增 5 个函数，用 `ast` 模块解析 chosen 代码的 AST：

| 函数                        | 职责                                                |
| --------------------------- | --------------------------------------------------- |
| `_ast_find_sql_defs()`      | 定位 SQL 字符串变量定义（直接赋值 / `text()` 包装） |
| `_ast_find_indirect_sql()`  | 处理 `_full_query()` 间接模式                       |
| `_ast_find_execute_calls()` | 定位参数化 `execute()` 调用                         |
| `_ast_build_concat_expr()`  | 根据 attack 类型构建拼接 SQL 表达式                 |
| `_ast_surgical_replace()`   | 行级手术替换：移除 SQL 赋值行 + 改写 execute 行     |
| `_ast_surgical_variant()`   | 主入口：编排上述步骤并验证结果                      |

**变换示例**：

```python
# chosen（安全）
sql = "SELECT * FROM orders WHERE status = %s"
with conn.cursor(DictCursor) as cur:
    cur.execute(sql, (value,))
    return cur.fetchall()

# ↓ _ast_surgical_variant(chosen, "string_concat")
# rejected（脆弱，结构完全同构）
with conn.cursor(DictCursor) as cur:
    cur.execute("SELECT * FROM orders WHERE status = '" + value + "'")
    return cur.fetchall()
```

### 优先级 2：正则策略（保留，作为 fallback）

原有的三种 `_try_variant_*` 函数**完整保留**。当 AST 变换无法处理某个 chosen 代码时（极少情况），回退到正则策略。

### 优先级 3：结构对齐回退（新增 `_dispatch_vulnerable_aligned`）

当 AST 变换和正则策略全部失败时，不再使用 `_dispatch_vulnerable()`（生成独立代码），而是使用新的 `_dispatch_vulnerable_aligned()`：

| 函数                             | 职责                                          |
| -------------------------------- | --------------------------------------------- |
| `_extract_imports()`             | 从 chosen 提取 import 语句                    |
| `_extract_main_func_info()`      | 从 chosen 提取函数名与参数签名                |
| `_detect_driver_from_code()`     | 检测 driver（pymysql / sqlite3 / sqlalchemy） |
| `_dispatch_vulnerable_aligned()` | 用提取的结构 + 脆弱 SQL 构造组装 rejected     |

## 验证

测试覆盖：
- **4 种安全模板**（`_safe_pymysql_fetch`、`_safe_sqlalchemy_select`、`_safe_sqlite`、`_safe_indirect_chain`）× **7 种攻击类型** = **28 个测试用例**
- **28/28 全部通过**（100%）：AST 解析通过 + 脆弱 SQL 模式命中
- **结构对齐验证**：import 100% 一致，函数签名 100% 一致（`_full_query` 辅助函数有意移除——因为 SQL 被内联）

| 测试维度             | 结果                                          |
| -------------------- | --------------------------------------------- |
| AST 变换成功率       | 28/28 (100%)                                  |
| `ast.parse()` 通过率 | 28/28 (100%)                                  |
| 脆弱模式命中率       | 28/28 (100%)                                  |
| Import 结构对齐      | 28/28 (100%)                                  |
| 函数签名对齐         | 28/28 (100%，`_full_query` 有意移除情况除外） |
| 对齐回退路径         | 28/28 (100%)                                  |
| 无未定义变量         | 28/28 (100%，含 SQLAlchemy dict 绑定修复）    |

## 二次修复（同日）

初次实现中发现两个子问题，已在本 changelog 日期下修复：

### 子问题 A：SQLAlchemy dict 绑定提取 key 而非 value（严重）

**位置**：`_ast_find_execute_calls()` L782-787

**Bug**：对 `session.execute(stmt, {"v": value})`，旧代码遍历 `second_arg.keys` 提取 `Constant("v")` 的 value `"v"`——这是 dict key（占位符名），并非 Python 变量。后续拼接生成 `session.execute(text("...'" + v + "'"))`，其中 `v` 未定义（参数名为 `value`）。

**修复**：改为遍历 `second_arg.values`，取第一个 `ast.Name` 节点的 `id`（即 `"value"`）。

### 子问题 B：间接模式 `_full_query()` 删除不完整（中）

**位置**：`_ast_surgical_replace()` 间接分支

**Bug**：搜索 `sql = _full_query()` 行时使用 `"_full_query()" in line` 子串匹配——会先命中 `def _full_query() -> str:` 行（索引更小），导致 `break` 后只删除了函数定义行，`sql = _full_query()` 赋值行残留。

**修复**：改为精确正则 `sql_var\s*=\s*func_name\s*\(` 匹配赋值语句，回退时才用 `"_full_query()" in line and "def " not in line`。

## 改动的文件

- `dataset/generate_expanded_dataset.py`
  - 新增 `_ast_find_sql_defs()`、`_ast_find_indirect_sql()`、`_ast_find_execute_calls()`、`_ast_build_concat_expr()`、`_ast_surgical_replace()`、`_ast_surgical_variant()`
  - 新增 `_extract_imports()`、`_extract_main_func_info()`、`_detect_driver_from_code()`、`_dispatch_vulnerable_aligned()`
  - 修改 `_vulnerable_variant_from_chosen()`：AST 优先级 > 正则 > None
  - 修改 `build_dpo_pairs()`：回退路径改用 `_dispatch_vulnerable_aligned()`
  - 修复 `_ast_find_execute_calls()`：dict 绑定从 `values` 而非 `keys` 提取变量名
  - 修复 `_ast_surgical_replace()`：间接模式精确匹配赋值行，避免误删 `def` 行
  - 保留 `_try_variant_sqlalchemy()`、`_try_variant_sqlite_pymysql_percent()`、`_try_variant_indirect_full_query()` 作为 fallback
