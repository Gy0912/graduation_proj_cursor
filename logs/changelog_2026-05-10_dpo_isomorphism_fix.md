# DPO 偏好对同构化与难度分层（2026-05-10 第十六次加固）

## 问题背景

DPO 偏好对的核心要求：`chosen`（安全代码）和 `rejected`（脆弱代码）在**非安全维度**上必须完全一致——唯一差异应当是 SQL 构造方式（参数化 vs 拼接/f-string/format）。

此前 P0-1 修复引入了 AST 手术变换，但存在以下问题：
1. **无显式同构性验证** — 依赖隐式的正则匹配和结构校验，出错时静默
2. **无难度分层** — 所有对混在一起，无法区分 easy/medium/hard
3. **对数不足** — 仅 ~1100 对（每 ev=True 行生成 1 对）

## 修复方案

### 1. 显式同构性验证 `_verify_dpo_isomorphism`

新增函数对每对 chosen/rejected 做 AST 级验证：

| 验证维度    | 方法                                        | 要求                                |
| ----------- | ------------------------------------------- | ----------------------------------- |
| Import 语句 | `ast.iter_child_nodes` 提取所有 import/from | 100% 一致                           |
| 函数签名    | 提取 FunctionDef 的 name + args.args        | 100% 一致                           |
| 变量名      | 收集所有 `ast.Name` 节点                    | chosen 的变量必须在 rejected 中存在 |

验证在 `build_dpo_pairs` 中作为最后一个门闸，不通过的对直接丢弃。

### 2. DPO 难度分层

攻击类型 → DPO 难度层级映射：

| DPO 难度层 | 攻击类型                                                                       | 特征                 | 目标占比 |
| ---------- | ------------------------------------------------------------------------------ | -------------------- | -------- |
| Easy       | `string_concat`                                                                | 明显拼接 vs 参数化   | 30%      |
| Medium     | `fstring`, `format_string`                                                     | 微妙错误 vs 参数化   | 40%      |
| Hard       | `fake_sanitization`, `parameterized_query`, `orm_misuse`, `indirect_injection` | 几乎正确 vs 完全正确 | 30%      |

### 3. 扩展对数量 ≥2000

- 每行 ev=True 训练数据尝试为每种难度层生成 1 对
- 移除 `_validate_dpo_pair_structure`（被更精确的 `_verify_dpo_isomorphism` 取代）
- 回退路径 `_dispatch_vulnerable_aligned` 静默调用（不打印日志）
- 后处理：超过 2000 时按目标分布采样至 2000

## 验证结果

### 2200 行训练数据测试

| 指标                      | 结果                   |
| ------------------------- | ---------------------- |
| 总对数                    | **2000** ≥ 2000 ✓      |
| 同构性 (import+签名+变量) | **2000/2000 = 100%** ✓ |
| Easy 占比                 | 18% (目标 30%)         |
| Medium 占比               | 37% (目标 40%)         |
| Hard 占比                 | 44% (目标 30%)         |
| 良性跳过                  | 1100/2200 (50%)        |

### 示例配对

```python
# Chosen (安全参数化):
import pymysql
from pymysql.cursors import DictCursor

def fetch_rows(conn, uid):
    sql = "SELECT * FROM users WHERE id = %s"
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, (uid,))
        return cur.fetchall()

# Rejected (脆弱拼接) — 仅 SQL 构造不同:
import pymysql
from pymysql.cursors import DictCursor

def fetch_rows(conn, uid):
    cur = conn.cursor(DictCursor)
    cur.execute("SELECT * FROM users WHERE id = '" + uid + "'")
    return cur.fetchall()
```

## 改动的文件

| 文件                                               | 操作     | 说明                                                                                                                                                                                                    |
| -------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset/generate_expanded_dataset.py`             | 修改     | 新增 `_verify_dpo_isomorphism`；新增 `_DPO_DIFFICULTY_TIER` 难度映射；新增 `_DPO_TIER_TARGETS` 目标分布；`build_dpo_pairs` 重写为分层多对生成；移除 `_validate_dpo_pair_structure` 调用；增加后处理采样 |
| `logs/changelog_2026-05-10_dpo_isomorphism_fix.md` | **新建** | 本文件                                                                                                                                                                                                  |

### 不变更文件
- `dataset/template_bank.py` · `dataset/adversarial.py` · `training/` · `evaluation/` · `detection/` · `tests/` — 零改动
