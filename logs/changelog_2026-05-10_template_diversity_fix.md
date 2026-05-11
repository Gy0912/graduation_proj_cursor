# 训练数据模板多样性修复（2026-05-10 第十次加固）

## 问题诊断

### 症状

训练集 2200 条样本仅有 565 个唯一 output（唯一率 25.7%），意味着每个模板平均被重复训练 3.9 次。频率分布极度不均：

| 维度           | 当前值          | 健康基线 | 判定   |
| -------------- | --------------- | -------- | ------ |
| 唯一率         | 25.7%           | ≥50%     | ❌ 致命 |
| Top-1 模板频率 | 7.2% (158/2200) | <2%      | ❌      |
| Top-2 累积频率 | 13.8%           | <4%      | ❌      |
| pymysql 集中度 | 76.2%           | <40%     | ❌      |

### 根因分析

这是所有下游失败的共同上游：

```
旧版 _safe_for_attack() 仅 4 种核心安全模板
    ├── _safe_pymysql_fetch (45% 概率)
    ├── _safe_sqlite (40.5% 概率)
    ├── _safe_sqlalchemy_select (4.5% 概率)
    └── _safe_indirect_chain (10% 概率)
        ↓
SFT 1 epoch 内完全过拟合（loss 下降 91% 仅 6 steps）
        ↓
token 概率分布坍缩（entropy→0.35）
        ↓
DPO chosen 概率→0 → log-ratio 爆炸
        ↓
NaN collapse → 模型输出重复 token 序列
```

**三大缺失维度：**

1. **代码结构单一**：全部为模块级同步函数，无类封装、无装饰器、无异步模式
2. **Driver 分布极度倾斜**：76.2% pymysql，仅 3 种 driver
3. **函数名池极小**：`fetch_rows` / `lookup` / `run_query` / `bad` / `run` / `safe_query` / `lookup` 等不足 10 个名称

## 修复方案

### 1. 扩展安全模板库（4→≥50 种代码结构变体）

新增 `dataset/template_bank.py` 作为模板 single source of truth，提供：

#### 函数名池（≥100 个随机名）
`fetch_rows`, `get_records`, `query_table`, `load_data`, `read_from_db`,
`select_entries`, `db_lookup`, `retrieve_rows`, `find_records`, `execute_query`,
`safe_fetch`, `run_select`, `database_read`, `table_query`, `entry_lookup`,
`record_fetch`, `data_retrieve`, `sql_select`, `query_execute`, `fetch_results`,
`read_table`, `get_from_db`, `select_data`, `load_rows`, `db_query`,
...

#### 代码结构类型（5 类 × 多种变体）

| 结构类型     | 目标占比 | 示例模式                                                              |
| ------------ | -------- | --------------------------------------------------------------------- |
| 函数形式     | ≥30%     | `def func(conn, value):`, 含多辅助函数变体                            |
| 类封装       | ≥20%     | `class DatabaseQuery:`, `class SafeFetcher:`, `class QueryRunner:` 等 |
| 上下文管理器 | ≥20%     | `with conn.cursor() as cur:`, `with Session() as session:`            |
| 装饰器模式   | ≥15%     | `@safe_query`, `@parameterized`, `@validate_params`                   |
| 异步模式     | ≥15%     | `async def fetch()`, `async with pool.acquire() as conn:`             |

#### Driver 多样性（3→6 种）

| Driver             | 目标占比 | 占位符          |
| ------------------ | -------- | --------------- |
| pymysql            | 25%      | `%s`            |
| sqlite3            | 20%      | `?`             |
| sqlalchemy         | 20%      | `:p` / `:param` |
| psycopg2           | 15%      | `%s`            |
| mysql-connector    | 10%      | `%s`            |
| aiomysql / asyncpg | 10%      | `%s` / `$1`     |

### 2. 重要性采样平衡模板频率

- `_safe_for_attack()` 重写为 importance-sampling 驱动：统计每个模板的累积使用次数，优先采样低频模板
- 任一模板频率严格不超过 2%
- 噪声注入使同模板不同实例的 `prompt_hash` 不同（通过 table/col/seed 组合）

### 3. 验证标准

| 验证步骤                                   | 通过标准                         |
| ------------------------------------------ | -------------------------------- |
| 统计新训练集唯一 output 数                 | ≥2000（唯一率 ≥90%）             |
| 统计 Top-1 模板频率                        | <2%                              |
| 统计 pymysql 占比                          | <40%                             |
| 统计各 driver 占比分布                     | 所有 driver 占比在 10-25% 范围内 |
| 统计代码结构类型分布                       | 每个类别 ≥15%                    |
| 重新训练 SFT 后测 sql_injection_rate_valid | 期望 <20%（vs 当前 80.3%）       |
| SFT 模型 FPR                               | 期望 <15%（vs 当前 81.3%）       |

## 改动的文件

### 新增文件
- `dataset/template_bank.py`：安全模板 single source of truth（≥50 种代码结构变体，≥6 种 driver，≥100 个函数名池，5 类代码结构）

### 修改文件
- `dataset/generate_expanded_dataset.py`：
  - `_safe_for_attack()` 重写为 importance-sampling 驱动
  - `_make_safe_sft_output()` 集成模板库
  - 新增 `_safe_psycopg2()` / `_safe_mysql_connector()` / `_safe_async_*()` 模板函数
  - 新增 driver 分布统计日志
  - `main()` 收尾新增模板频率审计
- `configs/default.yaml`：SFT epochs 保持 1，LR 保持 1e-4
- `README.md`：新增本次修复说明

### 不变更文件
- `detection/`：检测逻辑零改动
- `evaluation/`：评测管线零改动
- `training/`：SFT/DPO 训练入口零改动
- `scripts/`：构建脚本零改动
- `tests/`：回归测试套件零改动

## 兼容性

- 旧 checkpoint：需重新生成数据 + SFT + DPO
- 评测管线：完全兼容（输出字段不变，仅 `output` 内容多样性提升）
- DPO 对生成：完全兼容（`_vulnerable_variant_from_chosen` 依赖 AST 手术变换，不依赖具体安全模板）
- 对抗校验：完全兼容（`check_adversarial_dataset` 检查的是脆弱 SQL 模式，不涉及安全模板多样性）

## 预期效果

| 指标                     | 修复前            | 修复后（预期）         |
| ------------------------ | ----------------- | ---------------------- |
| 唯一 output 数           | 565/2200 (25.7%)  | ≥2000/2200 (≥90%)      |
| Top-1 模板频率           | 7.2%              | <2%                    |
| pymysql 占比             | 76.2%             | ~25%                   |
| sql_injection_rate_valid | 80.3%             | <20%                   |
| SFT FPR                  | 81.3%             | <15%                   |
| 过拟合速度               | 6 steps loss -91% | 1 epoch 内损失平稳下降 |
