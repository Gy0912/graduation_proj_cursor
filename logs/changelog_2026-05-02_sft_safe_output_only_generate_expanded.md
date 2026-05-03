# Changelog: SFT 数据集仅安全 `output`（`generate_expanded_dataset`）（2026-05-02）

## 修改

- **`dataset/generate_expanded_dataset.py`**
  - **删除**：ambiguous 分支、`_pick_subtle_output` 及全部 `_subtle_*`「细微不安全参考答案」生成路径。
  - **统一**：`build_one_sample` / `_fill_bucket_list` 的 salt 回退路径均通过 **`_make_safe_sft_output`**（`_hard_safe_reference` / `_safe_for_attack` + `_decorate_hard_output`）生成 `output`，与 **`expected_vulnerable` 元数据解耦**（该字段仍由 `label_queue` 控制约 50/50，**不再**改变 `output`）。
  - **校验**：新增 **`_output_valid_for_sft`**：`ast.parse` 通过且 **`contains_vulnerable_sql_pattern`** 为假；否则该次候选丢弃、循环重试（不写入不合格 `output`）。
  - **CLI**：`--num_samples` 下限改为 **20**（过小则多桶计数为 0，仅适合冒烟）；`eval_n` 改为按 `eval_ratio` 与 `num_samples` 动态计算，避免小 `num_samples` 时 `train_n` 为负。
  - **依赖**：从 `dataset.adversarial` 增加导入 **`contains_vulnerable_sql_pattern`**。

## 未修改

- **`build_dpo_pairs`**：逻辑未改；`chosen` 随训练行 `output` 变为恒安全；`rejected` 仍为 `_dispatch_vulnerable` 合成（后续若单独收紧 DPO 可另开任务）。

## 目的与影响

- **目的**：SFT 监督目标仅为**安全** Python，消除「正类标签 + 脆弱参考代码」对 next-token loss 的污染。
- **影响**：`data/train_expanded.json` 与评测行 `output` 均可假定无项目内 SQLi 模式；`expected_vulnerable` 保留供 FPR/FNR 等评测设计，但与参考答案代码风险脱钩。

## 手工建议

```powershell
.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py --num_samples 20 --eval_ratio 0.15 --seed 99
.\.venv\Scripts\python.exe -c "import ast,json; from pathlib import Path; from dataset.adversarial import contains_vulnerable_sql_pattern; rows=json.loads(Path('data/train_expanded.json').read_text(encoding='utf-8'));
for r in rows:
    ast.parse(str(r['output']).strip())
    assert not contains_vulnerable_sql_pattern(str(r['output']))[0]
print('ok', len(rows))"
```
