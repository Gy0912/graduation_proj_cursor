# Changelog: 移除 `generate_expanded_dataset` 中旧版结构化评测 prompt（2026-05-02）

## 修改

- **`dataset/generate_expanded_dataset.py`**
  - **删除** `template_prompt` 函数及其中的 `Output contract` 全文，以及对该分段式输出（含 marker 名与 fenced 代码要求）的硬编码。
  - **`to_eval_prompt_row`**：将原先 `template_prompt(...)` 改为 **`training_prompt(...)`**，使导出评测行的 `prompt` 与 SFT/DPO 所用前缀一致，仅消除**生成器侧**与训练之间的 prompt 结构冲突；不改变 `training_prompt` 实现、不改变 `build_one_sample` / ambiguous / DPO / `build_dpo_pairs` 等数据生成逻辑。

## 目的与影响

- **目的**：去掉与「仅 code / code_only_inference」等评测策略在**文案层**互相打架的旧版「必须输出 SECURITY/EXPLANATION/SAFE + fenced」契约，避免同一管线内两套 prompt 哲学并存。
- **影响**：此后新产出的 `data/eval_expanded.json` 中 `prompt` 字段形态与 `train_expanded` / `dpo_pairs` 的 prompt 对齐；仓库内已有 `eval_expanded.json` 需用户自行重新生成才会更新。
- **未触碰**：`evaluation/`、`training_prompt` 正文、`build_dpo_pairs`、检测与抽取逻辑。

## 验证

- 在 `dataset/generate_expanded_dataset.py` 内对字面量 `SECURITY WARNING` 做检索应为 **0** 处匹配。
