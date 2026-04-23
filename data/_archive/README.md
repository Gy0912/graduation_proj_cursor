# data/_archive

本目录保留**已弃用**的数据文件，仅用于审计与追溯。任何位于此目录的文件
**都不应**再被训练、评测或统计脚本直接引用。

## 文件说明

| 文件 / 子目录 | 来源 | 弃用原因 |
|---------------|------|----------|
| `combined_eval_legacy_unlabeled_2026-04-20.jsonl` | 原 `data/combined/eval.json`（由 `scripts/build_dataset.py` + `dataset/synthetic_sql.py` 写出） | Schema 仅为 `{id, prompt, meta}`，**不含 `expected_vulnerable`**；配合 `prompt_loader.py` 里 `bool(row.get("expected_vulnerable", False))` 的静默默认导致所有样本被判为非漏洞，TP=FN=0，评测指标数学上无效。 |
| `pre_2026-04-22_contaminated_sft/` | 2026-04-22 前由 `dataset/generate_expanded_dataset.py` 写出的 `data/*.json` 全套 | `expected_vulnerable=True` 样本的 `output` 字段是真·脆弱 SQL 代码，SFT 会把它当作 target 最小化——模型被手把手教会生成 SQL 注入。见 `logs/changelog_2026-04-22_adversarial_sft_training.md`。 |

## 当前权威评测集

新流程唯一使用：

```
data/combined/eval_fixed.json   # 由 scripts/build_eval_fixed.py 合并 data/generation/eval.json + data/fix/eval.json 生成
```

相关代码同步更新：
- `evaluation/prompt_loader.py::_normalize_sample` 缺标签直接 `ValueError`；
- `evaluation/evaluator.py::_assert_dataset_sanity` 启动前打印总数 / 正负样本数，单类立即 `RuntimeError`；
- 所有 `configs/*.yaml` 中 `files.eval_prompts` 指向 `data/combined/eval_fixed.json`。
