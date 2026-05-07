# P0-3 修复：add_special_tokens=False 破坏 DPO 序列边界

**日期**：2026-05-06  
**严重级别**：P0（训练-推理分布偏移）  
**影响范围**：`training/stable_dpo_trainer.py` — `_tokenize_one()` L110-112

## 问题

`_tokenize_one()` 对所有输入使用 `add_special_tokens=False`：

```python
prompt_ids = processing_class(prompt, add_special_tokens=False)["input_ids"]
chosen_ids = processing_class(chosen, add_special_tokens=False)["input_ids"]
rejected_ids = processing_class(rejected, add_special_tokens=False)["input_ids"]
```

StarCoder2 使用 `<|endoftext|>` 作为 BOS token。在预训练中每个文档以 BOS 开头。
`add_special_tokens=False` 意味着拼接后的 `prompt_ids + chosen_ids` 缺少 BOS 前缀。

### 影响

| 阶段     | BOS 状态                                           |
| -------- | -------------------------------------------------- |
| 推理     | 有 BOS（tokenizer 默认 `add_special_tokens=True`） |
| DPO 训练 | 无 BOS（`add_special_tokens=False`）               |

→ **训练-推理分布偏移**：序列起始处的 token 预测缺乏 BOS 上下文，模型学到错误的序列起始分布。

## 修复

```python
# 修复后：仅 prompt 前加 BOS，chosen/rejected 不加（避免中段重复）
prompt_ids = processing_class(prompt, add_special_tokens=True)["input_ids"]
chosen_ids = processing_class(chosen, add_special_tokens=False)["input_ids"]
rejected_ids = processing_class(rejected, add_special_tokens=False)["input_ids"]
```

- `prompt`：`add_special_tokens=True` → 序列以 BOS 开头
- `chosen`/`rejected`：`add_special_tokens=False` → 不在中段插入 BOS

拼接后：`[BOS, ...prompt..., ...chosen...]` — 与推理时的 tokenizer 默认行为一致。

## 二次修复（同日）：截断保留 BOS

初次修复中 `prompt_ids[-max_prompt:]` 和 `prompt_ids[-max_len:]` 取最后 N 个 token
的截断逻辑会在长 prompt 时丢弃开头的 BOS，使 P0-3 修复失效。

**修复**：两处截断改为 `[prompt_ids[0]] + prompt_ids[-(N-1):]`（保留 BOS，从右侧截断），
并对 `N≤1` 边界情况做保护。

## 改动的文件

- `training/stable_dpo_trainer.py`：`_tokenize_one()` — prompt 的 `add_special_tokens=True` + 截断保留 BOS

## 备注

SFT 训练路径（`training/sft_preprocess.py` L127-128）同样使用 `add_special_tokens=False`，
存在相同的分布偏移问题。若需要修复，方案相同：prompt 前加 BOS，completion 不加。
