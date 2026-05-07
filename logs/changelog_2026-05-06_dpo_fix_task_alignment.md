# P0-2 修复：fix 任务 chosen 与 vulnerable input 结构化对齐

**日期**：2026-05-06  
**严重级别**：P0（DPO 训练信号被工具偏好噪声污染）  
**影响范围**：`dataset/generate_expanded_dataset.py` — `build_one_sample()`、`_fill_bucket_list()`

## 问题

对于 `task="fix"` 的样本，`build_one_sample()` 和 `_fill_bucket_list()` 的工作流程：

1. `_dispatch_vulnerable()` 生成脆弱代码作为 input（如 SQLAlchemy `text()` 拼接）
2. `_make_safe_sft_output()` 从**零**生成安全代码作为 output/chosen（如 pymysql 参数化）

→ chosen 与 input 中的脆弱代码使用**不同的库、不同的函数名、不同的参数签名**。

### 实际影响

模型在 DPO 中学到的信号是「pymysql 参数化 > SQLAlchemy text() 拼接」，而非正确的「text() 参数化 > text() 拼接」。工具选择维度与安全选择维度被混淆，模型可能在 SQLAlchemy 上下文中不恰当地切换到 pymysql。

| 维度           | 旧版（P0-2 修复前）                                                  | 新版（P0-2 修复后）                                                 |
| -------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Input 脆弱代码 | `def bad(session, name): q = text("SELECT ...'" + ...)` (SQLAlchemy) | 同左                                                                |
| Chosen（安全） | `def fetch_rows(conn, value): sql = "SELECT ... %s"` (pymysql)       | `def bad(session, name): stmt = text("SELECT ... :p")` (SQLAlchemy) |
| 库             | pymysql ≠ SQLAlchemy ❌                                               | SQLAlchemy = SQLAlchemy ✅                                           |
| 函数名         | fetch_rows ≠ bad ❌                                                   | bad = bad ✅                                                         |

## 修复

### 新增 `_safe_fix_from_vulnerable(vuln_code, table, col)`

从脆弱代码生成结构同构的安全修复版本：

1. AST 解析脆弱代码，提取 imports、函数名、参数签名、driver
2. 从 SQL 中提取 table/column
3. 用 AST 分析找到被拼接到 SQL 中的用户输入参数
4. 构建同 driver、同函数名、同参数签名的参数化版本

```python
# 脆弱 input（SQLAlchemy orm_misuse）
from sqlalchemy import text

def bad(session, name: str):
    q = text("SELECT * FROM users WHERE user_id = '" + name + "'")
    return session.execute(q).fetchall()

# ↓ _safe_fix_from_vulnerable()

# 安全 output（同 driver、同函数名、参数化）
from sqlalchemy import text

def bad(session, name: str):
    stmt = text("SELECT * FROM users WHERE user_id = :p")
    return session.execute(stmt, {"p": name}).fetchall()
```

### 修改 `build_one_sample()` 和 `_fill_bucket_list()`

Fix 任务的 output 改用 `_safe_fix_from_vulnerable()`，回退到 `_make_safe_sft_output()` 仅当同构改写失败。

### 新增 `_extract_likely_param(vuln_code, tree)`

从脆弱代码的 `execute()` 调用 AST 中提取用户输入参数名。

## 验证

- **7 种攻击类型 × 3 种难度 = 21 个测试用例**
- **21/21**：driver 一致、函数名一致、ast.parse 通过、无脆弱模式命中
- 回退路径：`_safe_fix_from_vulnerable` 失败时自动回退到 `_make_safe_sft_output`

## 改动的文件

- `dataset/generate_expanded_dataset.py`
  - 新增 `_safe_fix_from_vulnerable()`
  - 新增 `_extract_likely_param()`
  - 修改 `build_one_sample()`：fix 任务 output 优先用同构改写
  - 修改 `_fill_bucket_list()`：同上
