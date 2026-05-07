# P0-4 修复：attack_type 标签与实际代码范式不一致

**日期**：2026-05-07  
**严重级别**：P0（元数据标签漂移导致后续分析误导）  
**影响范围**：`dataset/generate_expanded_dataset.py` — `build_dpo_pairs()` + `load_dpo_dataset`（两处训练脚本）

## 问题

`_safe_for_attack()` 和 `_hard_safe_reference()` 根据随机数选择安全代码实现：

```python
def _safe_for_attack(attack, table, col, rng):
    if attack == "indirect_injection":
        return _safe_indirect_chain(...) if rng.random() < 0.65 else _safe_pymysql_fetch(...)
    if attack == "orm_misuse":
        return _safe_sqlalchemy_select(...) if rng.random() < 0.5 else _safe_pymysql_fetch(...)
    if rng.random() < 0.45:
        return _safe_pymysql_fetch(...)
    if rng.random() < 0.9:
        return _safe_sqlite(...)
    return _safe_sqlalchemy_select(...)
```

由于随机性，DPO 对中 `attack_type` 可能与 `chosen` 实际代码范式无关：

| attack_type          | 概率 | chosen 实际范式                |
| -------------------- | ---- | ------------------------------ |
| `orm_misuse`         | 50%  | `pymysql`（而非 `sqlalchemy`） |
| `indirect_injection` | 35%  | `pymysql`（而非 `indirect`）   |
| `string_concat`      | 45%  | `pymysql`                      |

→ 任何基于 `attack_type` 的后续分析（按攻击类型统计 DPO 效果）都会产生误导。

## 修复

新增 `chosen_framework` 字段，由 `_detect_driver_from_code(chosen_body)` 检测 chosen 代码**实际使用**的驱动范式（`pymysql` / `sqlite3` / `sqlalchemy`）。

- `attack_type`：保持不变，反映 prompt 的攻击模式（仍是有价值的元数据）
- `chosen_framework`：新增，反映 chosen 代码实际范式（ground truth）

## 改动的文件

- `dataset/generate_expanded_dataset.py`
  - `build_dpo_pairs()`：`dpo.append` 新增 `chosen_framework` 字段
- `training/dpo_train.py`
  - `load_dpo_dataset()`：保留 `chosen_framework` 字段
- `training/train_qlora_dpo.py`
  - `load_dpo_dataset()`：保留 `chosen_framework` 字段
