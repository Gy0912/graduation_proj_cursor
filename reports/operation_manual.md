# 操作手册（当前版本）

## 1. 实验目标

- 针对 SQL 注入问题，验证大模型在：
  - 代码生成任务中的漏洞率变化
  - 代码修复任务中的修复成功率变化
  - 微调后基础能力保留情况

## 2. 环境准备（PowerShell）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/00_prepare_env.ps1
```

## 3. 生成运行配置

```powershell
Set-Location e:\graduation_proj
.\.venv\Scripts\python.exe scripts/prepare_default_run.py
.\.venv\Scripts\python.exe scripts/prepare_bandit_only_run.py
```

## 4. 数据构建

```powershell
.\.venv\Scripts\python.exe dataset/generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42
.\.venv\Scripts\python.exe scripts/build_dataset.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe scripts/build_eval_fixed.py
```

说明：
- `generate_expanded_dataset.py` 写 `data/combined/train.json`、`data/generation/*`、`data/fix/*`，并同步写出 `data/combined/eval_fixed.json`（内部已做缺标签 FAIL FAST）。
- `build_dataset.py` 仅产出 `dataset/*.jsonl` 的 SFT/DPO demo，不再触碰评测集。
- `build_eval_fixed.py` 是独立幂等工具：强校验 generation/fix 评测源 → 写 `data/combined/eval_fixed.json`。若任一样本缺 `expected_vulnerable` 或正负样本只有一类，立即报错并终止，**评测不会跑在无效数据上**。

## 5. 训练

```powershell
.\.venv\Scripts\python.exe training/train_lora_only.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/train_lora_sft.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/dpo_train.py --config configs/dpo.yaml
.\.venv\Scripts\python.exe training/train_qlora_only.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/train_qlora_sft.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/train_qlora_dpo.py --config configs/dpo.yaml
```

## 6. 评测（统一入口）

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model baseline
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_only
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_sft
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_dpo
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model qlora_only
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model qlora_sft
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model qlora_dpo --allow-missing-adapter
```

## 7. 汇总与可视化

```powershell
.\.venv\Scripts\python.exe scripts/compare_results.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe visualization/plot_compare_metrics.py --input outputs/compare_results.json --output-dir outputs/plots
```

说明：如果 PowerShell 禁止执行脚本，直接调用 `.\.venv\Scripts\python.exe` 是最稳定方案，无需 `Activate.ps1`。

输出：
- `outputs/*_results.json`
- `outputs/comparison_summary.json`
- `outputs/compare_results.json`
- `outputs/plots/*.png`

## 8. 指标解读（核心）

- `sql_injection_rate` 下降：SQL 注入风险降低
- `false_positive_rate` / `false_negative_rate`：反映检测偏差
- `precision` / `recall` / `f1`：与 `expected_vulnerable` 对齐后的分类效果

## 9. 评测数据集标签契约（2026-04-20 起强制）

评测入口 `evaluation/evaluate.py` 在启动时会：
1. 通过 `evaluation/prompt_loader.py` 加载 `files.eval_prompts`（默认 `data/combined/eval_fixed.json`）——任一样本缺 `expected_vulnerable` 即 `ValueError`。
2. 调用 `evaluation/evaluator.py::_assert_dataset_sanity` 打印：
   ```
   [Eval] Total samples: 600
   [Eval] Vulnerable: 300
   [Eval] Safe: 300
   ```
3. 若样本数为 0 或仅有一类（pos=0 或 neg=0）：`RuntimeError`。

这是最后一道门闸。若看到异常退出，请：
- 运行 `.\.venv\Scripts\python.exe scripts\build_eval_fixed.py` 重建评测集；
- 或核对 `data/generation/eval.json` / `data/fix/eval.json` 中 `expected_vulnerable` 标签是否齐全且为 bool。

旧版 `data/combined/eval.json` 已归档到 `data/_archive/`，**不要再把配置里的 `files.eval_prompts` 改回该路径**。

## 10. 备注

- 如需更换基座模型，修改 `configs/default.yaml` 中 `model.base_model`，再重新执行 `prepare_default_run.py`。
- 统一术语：训练入口在 `training/`，评测入口固定为 `evaluation/evaluate.py`。
