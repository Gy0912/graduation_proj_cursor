# Changelog: DPO 偏好对语义对齐（2026-05-02）

## 背景

`build_dpo_pairs` 原先对每条训练样本用 **随机** `_pick_table_col(rng)` 再调用 `_dispatch_vulnerable` 生成 `rejected`。`chosen` 来自该行安全 `output`，二者虽共享同一 `prompt`，但 **SQL 目标（表/列/代码骨架）可完全不一致**，DPO 信号退化为「任意安全片段 > 任意脆弱片段」，不利于学习「在同一任务上偏好安全实现」。

## 修改摘要

### `dataset/generate_expanded_dataset.py`

1. **`schema_table` / `schema_column`**  
   - `build_one_sample` 与 `_fill_bucket_list` 盐补路径写入每条训练行，与生成 `input` 时使用的表、列一致。  
   - **目的**：DPO 与审计可显式核对 schema，且 `_infer_schema_from_row` 可优先读字段（再回退解析 `input`）。

2. **`_infer_schema_from_row`**  
   - 从行字段或 `input` 文本解析 `(table, column)`；失败则 **FAIL FAST**（避免静默错位）。

3. **`_vulnerable_variant_from_chosen` + `_try_variant_*`**  
   - 在 **不改变** `instruction`/`input`/`prompt` 的前提下，将安全 `chosen` 改写为 **同构脆弱**实现：  
     - pymysql/sqlite：去掉 `sql = ...` 占位行，改为 `cur.execute("SELECT...'" + param + "'")` 等，命中 `execute_plus_concat` / `fstring_sql` 等规则。  
     - SQLAlchemy：合并 `stmt`+`return session.execute(...)` 为带 `text(...)` 内拼接或 `session.execute(f"...")` 的单条 `return`。  
     - `_full_query` 间接模板：删除 `_full_query` 与 `sql = _full_query()`，保留 `with` 块内联 `cur.execute(...)`。  
   - **`format_string` 攻击族**：采用 `"...".format` 与 `+` 组合以满足 `contains_vulnerable_sql_pattern`（`concat_plus_sql` 的 `'" ... '+` 形态对 `sql = "..." +` 不敏感，故优先走 `execute(...)` 内拼接）。

4. **`build_dpo_pairs`**  
   - 使用 `_vulnerable_variant_from_chosen` 替代 `_dispatch_vulnerable`。  
   - 校验：`chosen` 不得命中脆弱模式；`rejected` 必须命中；二者 `ast.parse`。  
   - DPO 行增加 `schema_table` / `schema_column`。

5. **`main`**  
   - 生成结束后打印 **3 条** DPO 抽样（prompt 前缀、chosen/rejected 头部、schema），便于人工确认「同题、仅脆弱性差异」。

## 明确未改动的部分（契约）

- **SFT**：`train_expanded.json` 的 `output` 仍仅来自 `_make_safe_sft_output`，**不会**把脆弱目标写回 SFT。  
- **DPO 训练脚本**：`training/dpo_train.py` 等仍只消费 `prompt` / `chosen` / `rejected` JSONL 字段；新增 `schema_*` 为可选元数据，旧脚本忽略即可。

## 影响

- 重新运行 `python dataset/generate_expanded_dataset.py ...` 后，`data/dpo_pairs.json` 中偏好对 **语义一致**，更利于 TRL `DPOTrainer` 学习「同任务安全 > 脆弱」。  
- 生成耗时与失败率：若某 `attack_type` 与安全模板组合无法产出可检出脆弱变体，生成器会 **raise**（此前几乎总成功）；当前矩阵已在本地对全 `attack_type` × `difficulty` 随机抽样通过。

## 相关文档

- `README.md`：新增「DPO semantic alignment」小节与顶栏摘要。  
