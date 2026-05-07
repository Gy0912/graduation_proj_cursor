# P1-6 修复：max_length 截断可使 chosen/rejected 为空

**日期**：2026-05-06  
**严重级别**：P1（超长 prompt 样本零有效训练信号 + 浪费计算）  
**影响范围**：`training/stable_dpo_trainer.py` — `_tokenize_one()` + `_prepare_dataset()`

## 问题

当 `max_len` 截断中 `budget <= 0`（prompt 本身超过 max_length）：

```python
budget = max_len - len(prompt_ids)  # = 0
chosen_ids = chosen_ids[:0]         # → []（空）
rejected_ids = rejected_ids[:0]     # → []（空）
```

后果：

| 后果                                     | 详情                           |
| ---------------------------------------- | ------------------------------ |
| `chosen_labels` 全 `-100`                | loss = 0，无梯度               |
| `chosen_input_ids == rejected_input_ids` | DPO log-ratio = 0，梯度 = 0    |
| 浪费 GPU 算力                            | 样本被处理但不产生有效训练信号 |

## 修复

### 1. `_tokenize_one()` 守卫

```python
if len(chosen_ids) == 0 or len(rejected_ids) == 0:
    return None  # 标记无效
```

### 2. `_prepare_dataset()` 过滤 + 日志

```python
tokenized_rows = [t for t in tokenized_rows if t is not None]
# 打印丢弃比例 warning
# 全丢弃时 raise ValueError
```

## 改动的文件

- `training/stable_dpo_trainer.py`
  - `_tokenize_one()`：空 chosen/rejected 返回 `None`
  - `_prepare_dataset()`：过滤 `None`，warning 日志，全丢弃 ValueError
