# DPO tokenization pipeline 彻底重写（2026-05-08）

**严重级别**：P0（NaN loss / 梯度爆炸 / 模型 collapse）  
**影响范围**：`training/stable_dpo_trainer.py` — `_prepare_dataset()`

## 根因分析

旧版 `StableDPOTrainer._prepare_dataset()` 为规避 datasets.map 多进程崩溃，
手工实现了一套"简化版 DPO tokenization"：

```python
# 旧版（已废弃）
prompt_ids = processing_class(prompt, add_special_tokens=True)["input_ids"]
chosen_ids = processing_class(chosen, add_special_tokens=False)["input_ids"]
rejected_ids = processing_class(rejected, add_special_tokens=False)["input_ids"]
chosen_input_ids = prompt_ids + chosen_ids       # 手工拼接
chosen_labels = [-100] * len(prompt_ids) + chosen_ids
...
return {"chosen_ids": chosen_ids, ..., "prompt_ids": prompt_ids, ...}
```

### 为什么手工简化版必然失败

| 问题                     | 旧版行为                                                        | 后果                                                                                                                                                                             |
| ------------------------ | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EOS 缺失**             | chosen/rejected 未追加 EOS token                                | 模型学不到序列终止信号，token 概率分布异常                                                                                                                                       |
| **边界错误**             | prompt 和 chosen 分别 tokenize 后拼接                           | tokenizer 在拼接边界 whitespace 处理不一致，HF Trainer 内部无法正确对齐 prompt/completion                                                                                        |
| **completion mask 错乱** | 手工 `[-100]*len(prompt)+chosen_ids`                            | TRL 的 DPODataCollator 期望从 `prompt_ids`/`chosen_ids` 长度**自动**构建 mask——旧版提供了预构建的 `chosen_labels`/`chosen_attention_mask`，导致 collator 重复处理，mask 区域偏移 |
| **字段名混乱**           | 返回 `chosen_ids` 为 full `prompt+chosen`，而非 completion-only | DPODataCollator 内部 `prompt_ids + chosen_ids` 变成 `prompt + (prompt+chosen)`，序列长度翻倍，OOM / NaN                                                                          |
| **BOS 重复**             | prompt 与 chosen 由不同 `add_special_tokens` 控制               | 合并 `prompt+chosen` 的 tokenization 语义不一致                                                                                                                                  |

### 症状链

```
EOS丢失 → completion无终止 → logprob尾部异常
    → policy偏离参考分布无约束 → 梯度幅值飙升
        → step~30处梯度爆炸 → logits NaN
            → 模型collapse为重复逗号
```

## 修复方案

**核心原则**：不再手工构建 tokenization pipeline。严格复制 TRL 原生 tokenize_fn 行为，
仅将 `dataset.map()` 替换为主进程 for 循环。

### 新的 _tokenize_one 流程

```
1. 追加 EOS token → chosen/rejected（TRL 原生 add_eos）
2. tokenize(prompt, add_special_tokens=True) → prompt_ids
3. tokenize(prompt+chosen, add_special_tokens=True) → full_chosen_ids
4. chosen_ids = full_chosen_ids[len(prompt_ids):]（TRL 原生拆分）
5. 同上处理 rejected
6. 返回 {"prompt_ids","chosen_ids","rejected_ids"}（DPODataCollator 契约）
```

### 数据集字段契约

| 字段           | 含义                        | 由谁消费                                         |
| -------------- | --------------------------- | ------------------------------------------------ |
| `prompt_ids`   | 仅 prompt（含 BOS）         | DPODataCollator                                  |
| `chosen_ids`   | **仅 completion**（含 EOS） | DPODataCollator 内部 `prompt_ids + chosen_ids`   |
| `rejected_ids` | **仅 completion**（含 EOS） | DPODataCollator 内部 `prompt_ids + rejected_ids` |

completion_mask、attention_mask、labels **全部由 DPODataCollator 自动构建**——不在 `_prepare_dataset` 中手工生成。

### 调试日志

新增训练启动日志：

```
DPO tokenization stats: total=1536 kept=1536 dropped_empty_comp=0 dropped_trunc=0 eos_appended=3072 max_lens=(prompt=211 chosen=78 rejected=82 total=293)
```

## 改动文件

- `training/stable_dpo_trainer.py`
  - `_prepare_dataset()` 彻底重写：EOS 追加 → 合并 tokenize → 拆分 → 返回 collator 契约字段
  - 新增 debug stats 日志
  - `_safe_tokenize_warning_context` / `compute_loss` 等其他方法不变
