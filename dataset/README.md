# dataset/

## 当前主数据流程

- 主入口脚本：`dataset/generate_expanded_dataset.py`
- 主要输出位置：
  - `data/combined/train.json`
  - `data/combined/eval_fixed.json`（权威评测集：由 generation+fix 拼合，带完整 `expected_vulnerable`）
  - `data/generation/{train,eval}.json` / `data/fix/{train,eval}.json`
  - `data/eval_expanded.json`
  - `data/train_expanded.json`
  - `data/dpo_pairs.json`

## 兼容数据流程

- `scripts/build_dataset.py`：按配置生成 `dataset/*.jsonl`（如 `sft_train.jsonl`、`sft_val.jsonl`、`dpo_train.jsonl`）。**此脚本不再写任何评测集**。
- `scripts/build_eval_fixed.py`：独立幂等地合并 `data/generation/eval.json` + `data/fix/eval.json` → `data/combined/eval_fixed.json`，带强校验。
- `dataset/generate_sql_security_dataset.py`：旧版小规模生成器，输出 `dataset/sql_security_dataset.json`，当前不作为主流程默认入口。

## 与训练/评测衔接

- `training/train_lora_sft.py`：读取配置中的 `files.train_sft_json`（默认 `data/combined/train.json`）。
- `training/dpo_train.py`：读取配置中的 `files.dpo_pairs`（默认 `data/dpo_pairs.json`）。
- `evaluation/evaluate.py`：读取配置中的 `files.eval_prompts`（默认 `data/combined/eval_fixed.json`）。任一样本缺 `expected_vulnerable` 立即 `ValueError`；空集或单类数据集立即 `RuntimeError`。
