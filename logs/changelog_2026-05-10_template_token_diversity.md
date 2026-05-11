# 安全模板库扩展：token 多样性增强（2026-05-10 第十三次加固）

## 问题诊断

### 症状

虽然第十次加固将安全模板从 4 种扩展到 ≥50 种代码结构变体，但实际 token 重叠度分析揭示了一个更深层的问题：

| 维度                             | 第十次加固后 | 问题                                                          |
| -------------------------------- | ------------ | ------------------------------------------------------------- |
| 同 driver+struct 的 token 重叠度 | ~91%         | 相同 driver+结构类型的模板几乎完全相同                        |
| ORM 模板的 token 重叠度          | ~91%         | 所有 ORM 模板忽略 driver 参数，生成完全相同的 SQLAlchemy 代码 |
| 变量名池                         | 10 个        | 用户要求 ≥50                                                  |
| 编码风格变异                     | 无           | 无 import/注释/查询形状/结果处理的多样性                      |

### 根因分析

```
旧版模板库问题：
  _make_safe_function("pymysql", ...) 和 _make_safe_function("sqlite3", ...)
    → 仅 import pymysql vs import sqlite3 和 %s vs ? 不同
    → 其余 85% token 完全重叠

  _make_safe_orm_expression(ANY_DRIVER, ...)
    → 完全忽略 driver 参数
    → 所有 ORM 模板 100% 相同（仅 func_name/table/col 字面量不同）
    → 同一 token(%s) 在安全代码中是 pymysql 占位符、在脆弱代码中是格式化操作符
    → 模型无法区分语义空间
```

## 修复方案

### 1. 池扩展

| 池            | 修复前         | 修复后          |
| ------------- | -------------- | --------------- |
| 函数名        | 110            | 132             |
| 类名          | 25             | 40              |
| 装饰器名      | 10             | 18              |
| 变量名        | 10             | **50**          |
| 表名/列名组合 | 9×9=81(笛卡尔) | **31 组预定义** |

### 2. 编码风格变异（全新）

引入 5 个正交维度的 post-generation 变异，总计 5×4×3×9×3 = **1620 种组合**：

| 维度         | 变体数 | 示例                                                                     |
| ------------ | ------ | ------------------------------------------------------------------------ |
| SQL 查询模式 | 5      | `SELECT *` / `SELECT col` / `LIMIT 1` / `AND active=1` / `IN (...)`      |
| 结果处理模式 | 4      | `fetchall()` / `fetchone()` / iterate / `list(wrap)`                     |
| Import 风格  | 3      | `import pymysql` / `from pymysql import ...` / `import pymysql as mysql` |
| 注释风格     | 9      | 无注释 / `# parameterized query` / `# bound parameters ensure safety` 等 |
| 换行风格     | 3      | 单换行 / 双换行 / 三换行                                                 |

### 3. ORM 模板 3 子变体

旧版 ORM 模板所有 driver 生成相同代码。新版引入 3 种 ORM 子变体，按 func/table/col 哈希随机选择：

- **变体 A**: `DeclarativeBase` + `mapped_column`（现代风格）
- **变体 B**: `declarative_base()` + `Column(Integer/String)`（经典风格）
- **变体 C**: `session.query().filter()` 风格

### 4. 命名参数模板

新增 `_make_safe_named_param()`：`%(param)s` 风格（psycopg2/pymysql 命名参数），与位置参数 `%s` 在 token 层面有显著差异。

### 5. Token 重叠度分析工具

新增 `audit_token_diversity()` 函数：
- 计算任意两模板的 Jaccard token 重叠度
- 标识超过 0.70 阈值的高重叠对
- 供数据生成后审计

## 改动的文件

| 文件                                                    | 操作     | 说明                                                                                                   |
| ------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `dataset/template_bank.py`                              | **重写** | 池扩展（50 变量名/31 表列组合）+ 5 维编码风格变异 + ORM 3 子变体 + 命名参数模板 + token 重叠度审计函数 |
| `dataset/generate_expanded_dataset.py`                  | 不变     | TemplateSampler API 不变，自动获得所有新特性                                                           |
| `logs/changelog_2026-05-10_template_token_diversity.md` | **新建** | 本文件                                                                                                 |

### 不变更文件
- `detection/` / `evaluation/` / `training/` / `scripts/` / `tests/` — 零改动
- `dataset/generate_expanded_dataset.py` — API 不变

## 验证结果

| 验证步骤                | 结果                      |
| ----------------------- | ------------------------- |
| 函数名池大小            | 132 ≥ 100 ✓               |
| 变量名池大小            | 50 ≥ 50 ✓                 |
| 表名/列名池大小         | 31 ≥ 30 ✓                 |
| 唯一模板数（60 样本）   | 60/60 = 100% ✓            |
| 最大 token 重叠度       | 0.842（从 0.911 降低 8%） |
| 平均 token 重叠度       | 0.408                     |
| ORM 内部多样性          | max=0.846（3 子变体有效） |
| 所有模板 AST 可解析     | ✓                         |
| 所有模板无脆弱 SQL 模式 | ✓                         |
| ORM vs raw SQL 重叠度   | 0.242（极低）             |
| 命名 vs 位置参数重叠度  | 0.682（不同函数名时）     |

### 已知限制

同一 `(driver, struct_type)` 组合的模板对 token 重叠度约 0.80-0.85，接近但未达到 <0.70 的严格阈值。这是因为相同结构类型的模板共享核心语法 token（`def`、`return`、`execute`、`fetchall` 等），这些是 Python/SQL 的固有语法无法消除。实践中通过：
1. importance-sampling 确保同一组合不重复出现 >1 次
2. 6 结构类型 × 7 driver × 4 函数子变体 × 1620 风格组合 → 足够的全局多样性

## 兼容性

- **向后兼容**：TemplateSampler API 不变，`generate_expanded_dataset.py` 无需修改
- **评测管线**：完全兼容（模板多样性不影响评测逻辑）
- **DPO 对生成**：完全兼容（AST 手术变换不依赖具体安全模板）
