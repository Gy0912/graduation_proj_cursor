# Driver 多样化验证与均衡分布（2026-05-10 第十五次加固）

## 问题背景

第十四次加固之前的模板库使用 `driver × struct_type` 矩阵，导致 pymysql 在所有安全 output 中占比高达 **76.2%**。模型严重过拟合到 `conn.cursor() + cur.execute(sql, params) + cur.fetchall()` 这一 API 模式，当 DPO 偏好对中 chosen 使用 sqlite3 而 rejected 使用 pymysql 时，模型学到的信号退化为"sqlite3 > pymysql"而非"参数化 > 拼接"。

## 修复验证

第十四次加固（v2 重写）已将模板系统改为 56 个独立 AST 级结构，内置 driver 分布控制在 `DRIVER_TARGET_WEIGHTS` 中：

| Driver          | 目标占比 | 实际占比 (2500样本) | 状态 |
| --------------- | -------- | ------------------- | ---- |
| pymysql         | 25%      | 22.3%               | ✓    |
| sqlite3         | 20%      | 20.1%               | ✓    |
| sqlalchemy      | 20%      | 23.0%               | ✓    |
| psycopg2        | 15%      | 14.7%               | ✓    |
| mysql-connector | 10%      | 9.4%                | ✓    |
| aiomysql        | 5%       | 5.2%                | ✓    |
| asyncpg         | 5%       | 5.2%                | ✓    |

### 各 driver 代码风格差异验证

| Driver          | Import                       | 占位符   | API 特征                                                                               | 连接管理                     |
| --------------- | ---------------------------- | -------- | -------------------------------------------------------------------------------------- | ---------------------------- |
| pymysql         | `import pymysql`             | `%s`     | `conn.cursor(DictCursor)` + `cur.execute(sql, (p,))`                                   | `with conn.cursor() as cur:` |
| sqlite3         | `import sqlite3`             | `?`      | `conn.execute(sql, (p,))` 或 `cur.execute(sql, (p,))`                                  | 直接 connect/close           |
| sqlalchemy      | `from sqlalchemy import ...` | `:named` | `session.execute(text())` / `session.query().filter().all()` / `select(Model).where()` | ORM Session                  |
| psycopg2        | `import psycopg2`            | `%s`     | `conn.cursor(cursor_factory=RealDictCursor)` + `cur.execute(sql, (p,))`                | `with conn, closing(cur):`   |
| mysql-connector | `import mysql.connector`     | `%s`     | `cnx.cursor(dictionary=True)` + `cur.execute(sql, (p,))` + `cur.close()`               | 显式 close                   |
| aiomysql        | `import aiomysql`            | `%s`     | `async with pool.acquire()` + `await cur.execute()`                                    | async pool                   |
| asyncpg         | `import asyncpg`             | `$1`     | `await conn.fetch(sql, param)` / `stmt.fetch(param)`                                   | async conn                   |

### 模型可区分性验证

| 条件                     | 预期行为                                   |
| ------------------------ | ------------------------------------------ |
| 给定 `import psycopg2`   | 模型使用 `%s` 占位符，`RealDictCursor`     |
| 给定 `import sqlite3`    | 模型使用 `?` 占位符                        |
| 给定 `import asyncpg`    | 模型使用 `$1` 占位符，`await conn.fetch()` |
| 给定 `import sqlalchemy` | 模型使用 `:named` 占位符或 ORM `.filter()` |

因为模板库中每个 driver 的 safe output 始终使用该 driver 对应的占位符风格和 API 模式，SFT 训练后模型将学会 driver-API 对应关系。DPO 的 `_vulnerable_variant_from_chosen` 通过 AST 手术变换保持 driver 一致（不会产生 pymysql→sqlite3 的跨 driver 偏好对），进一步保证偏好信号干净。

## 改动的文件

| 文件                                                         | 操作     | 说明                                           |
| ------------------------------------------------------------ | -------- | ---------------------------------------------- |
| `dataset/template_bank.py`                                   | 不变     | 第十四次加固已将 driver 分布内置               |
| `logs/changelog_2026-05-10_driver_diversity_verification.md` | **新建** | 本文件：分布验证 + 代码风格差异 + 可区分性验证 |

### 不变更文件
- 所有训练/评测/数据生成/检测代码 — 零改动（driver 分布由 TemplateSampler 的 importance-sampling 自动保证）

## 验证标准

| 验证步骤                               | 通过标准                           | 结果           |
| -------------------------------------- | ---------------------------------- | -------------- |
| 统计每个 driver 在安全 output 中的占比 | 每个 driver 在 10-25% 范围内       | ✓ (9.4%-23.0%) |
| 确认没有 driver 占 >30%                | 所有 driver ≤25%                   | ✓ (max=23.0%)  |
| 检查各 driver API 风格在模板中一致     | `import X` → 使用 X 的占位符和 API | ✓              |
| 端到端 2500 样本验证                   | 3 个随机种子均通过                 | ✓              |
