# Changelog — 2026-04-20 评测数据集标签强制约束

## 摘要

修复一个**关键评测 Bug**：此前 `data/combined/eval.json` 不包含 `expected_vulnerable` 字段，而 `evaluation/prompt_loader.py` 使用 `bool(row.get("expected_vulnerable", False))` 静默默认为 False，导致所有样本被视为 non-vulnerable。结果：TP = FN = 0，Recall / F1 数学上无效，整个分类评估失去意义。

本次更新切换到**强标签契约**，跨 5 层入口串联 FAIL FAST 门闸，让评测**根本不可能**在缺标签或单类的数据集上运行。

---

## 1. 新增

### `scripts/build_eval_fixed.py`（新脚本）

- 独立幂等的合并工具：
  - 输入：`data/generation/eval.json`（300 条）+ `data/fix/eval.json`（300 条）
  - 输出：`data/combined/eval_fixed.json`（600 条，pos=300 / neg=300）
- 对每条样本强校验：`id`、`task_type`、`instruction`、`expected_vulnerable`（必须是 bool）、`vulnerability_type`、`difficulty`。
- 按 `id` 去重；当 id 缺失时按 (instruction, input_code) 复合键去重。
- 合并后再次校验正负样本数均 > 0；否则 `RuntimeError`。
- 打印 kept/duplicates/total/pos/neg 统计。

### `data/combined/eval_fixed.json`（新数据文件）

- 新的、唯一的权威评测集。
- 样本字段：`id`、`task_type`、`instruction`、`input_code`、`expected_vulnerable`、`vulnerability_type`、`difficulty`。
- 规模：600 条，pos/neg = 300/300（由合并脚本强制校验）。

### `data/_archive/`（新归档目录）

- `combined_eval_legacy_unlabeled_2026-04-20.jsonl`：原 `data/combined/eval.json`（274 条 JSONL，schema 只有 id/prompt/meta，无 `expected_vulnerable`）。
- `README.md`：说明弃用原因与当前权威评测集的位置。

---

## 2. 修改

### `evaluation/prompt_loader.py`

- **强制校验 `expected_vulnerable`**：`_normalize_sample` 中 `expected_vulnerable not in row` 时立即 `ValueError("Missing expected_vulnerable in evaluation sample ...")`；值非 bool 也拒绝。
- **连带严格化其它字段**：`vulnerability_type` 必须非空字符串，`difficulty` / `task_type` 必须显式存在且非空；无法构造 prompt（prompt/instruction/input 三者皆空）时拒绝。
- 删除所有 `row.get(..., False)` / `row.get(..., "unknown")` 式的静默回退。
- JSON 数组支路里原先允许把 `str` 直接当成 prompt，现在一律要求 `dict`（避免无标签元素绕过）。
- JSONL 空文件改为 `ValueError`（之前返回空列表容易配合单类数据集伪装）。

### `evaluation/evaluator.py`

- 新增 `_assert_dataset_sanity(samples)`：
  - 空列表 → `RuntimeError`。
  - 每条样本缺 `expected_vulnerable` 或非 bool → `RuntimeError`。
  - 正类或负类样本数为 0 → `RuntimeError("Invalid evaluation dataset: only one class present (pos=..., neg=...)")`。
  - 打印三行：`[Eval] Total samples: X` / `[Eval] Vulnerable: Y` / `[Eval] Safe: Z`。
- 在 `run_eval_always_safe` 与 `run_eval_on_prompts` 的**第一行**调用 `_assert_dataset_sanity(samples)`。
- 新增 `_require_expected_vulnerable(src)`：替换所有 `bool(src.get("expected_vulnerable", False))`，缺键 `KeyError`，非 bool `TypeError`。
- `_per_sample_from_detection` 以及 `run_eval_on_prompts` 中「无效抽取分支」的样本记录构造现在都走新的严格读取。

### `dataset/research_schema.py`

- `to_research_record(..., include_output=False)`（评测路径）现在缺 `expected_vulnerable` 或非 bool 直接 `ValueError`；训练路径仍允许 `bool(row.get(..., False))` 兼容旧数据。
- `write_research_splits`：
  - **不再写 `data/combined/eval.json`**；改为写 `data/combined/eval_fixed.json`。
  - 写出前调用新增的 `_assert_eval_rows_labeled`：逐条校验 + 正负双类检查。
- `train.json` 与 `generation/fix` 的分裂输出保持不变。

### `scripts/build_dataset.py`

- **移除** 对 `files.eval_prompts` 的 JSONL 覆盖（以前会把无标签的合成 JSONL 写到 `data/combined/eval.json`，这是 bug 的源头之一）。
- 现在只写 `files.train_sft`、`files.val_sft`、`files.train_dpo`；对 `files.eval_prompts` 打印 `[skip]` 说明。

### `scripts/migrate_dataset_to_research_schema.py`

- `_normalize_eval_row`：缺 `expected_vulnerable` 或类型非 bool 立即 `ValueError`，不再默认 False。
- 文档说明从旧的 `data/combined/eval.json` 更新为 `data/combined/eval_fixed.json`。

### `scripts/run_thesis_pipeline.py`

- 在 `build_dataset.py` 之后、`baseline` 评测之前插入一步：`run([py, "scripts/build_eval_fixed.py"])`，确保流水线首次运行就能产出权威评测集。

### `configs/default.yaml` / `configs/default_run.yaml` / `configs/default_bandit_only_run.yaml`

- `files.eval_prompts: data/combined/eval.json` → `files.eval_prompts: data/combined/eval_fixed.json`。
- `configs/default.yaml` 增加注释：说明新路径来源、旧路径已归档。

### 文档

- `README.md`：新增「评测数据集标签强制约束（2026-04-20 修复）」段落；PowerShell 命令加入 `scripts\build_eval_fixed.py`；项目结构树更新；示例样本结构展示。
- `PROJECT_STRUCTURE.md`：data/ 与 scripts/ 段落更新；标记新增脚本与归档目录。
- `reports/operation_manual.md`：第 4 节数据构建加入 `build_eval_fixed.py`；新增第 9 节「评测数据集标签契约」。
- `scripts/README.md` / `dataset/README.md`：同步新流程。
- `data/_archive/README.md`：新增，说明归档原因。

---

## 3. 删除 / 归档

- **`data/combined/eval.json`**：移动至 `data/_archive/combined_eval_legacy_unlabeled_2026-04-20.jsonl`（JSONL 后缀对齐其真实格式）；不再被任何入口引用。

---

## 4. 影响与收益

### 立即的正确性收益

- 修复前：TP=FN=0，`precision / recall / f1 / FPR / FNR` 在代码层被兜底成 `0.0`，所有评测结果「无声失效」。
- 修复后：
  - 权威评测集 600 条，标签平衡（pos=neg=300）；
  - 混淆矩阵四格（TP/FP/TN/FN）都能被有意义地填满；
  - `by_attack_type_metrics`、`per_detector_vs_expected` 的分层结果也得到真实可比的分母。

### 工程级收益

- **五重保险**，让无效数据集根本进不了指标聚合：
  1. `build_eval_fixed.py`（源合并）→
  2. `research_schema.write_research_splits`（写出前）→
  3. `prompt_loader.load_eval_prompts`（加载时）→
  4. `evaluator._assert_dataset_sanity`（评测启动时）→
  5. `evaluator._require_expected_vulnerable`（per-sample 构造）。
- 任何一层出错都会抛出带上下文的异常（含 sample id / idx），调试成本显著下降。
- `build_dataset.py` 不再能覆盖评测集，杜绝了最初 bug 的结构性成因。

### 向后兼容性

- **无**。本次明确采用 FAIL FAST 策略：
  - 任何依然依赖旧 `data/combined/eval.json` 的脚本必须切换到 `data/combined/eval_fixed.json`；
  - 任何携带缺标签样本的数据源将立刻报错。
- 这是刻意的设计：旧行为会把错误结果伪装成合法指标发布，违反研究复现性。

---

## 5. 验证（已通过）

### 5.1 正向路径

```powershell
.\.venv\Scripts\python.exe scripts\build_eval_fixed.py
# [build_eval_fixed] total=600 pos=300 neg=300
# [build_eval_fixed] wrote .../data/combined/eval_fixed.json
```

```powershell
.\.venv\Scripts\python.exe -c "from evaluation.prompt_loader import load_eval_prompts; from evaluation.evaluator import _assert_dataset_sanity; s=load_eval_prompts('data/combined/eval_fixed.json'); _assert_dataset_sanity(s)"
# Detected JSON format
# [Eval] Total samples: 600
# [Eval] Vulnerable: 300
# [Eval] Safe: 300
```

### 5.2 逆向路径（都应 fail fast）

| 场景 | 位置 | 期望异常 | 结果 |
|------|------|---------|------|
| 加载旧无标签 JSONL | `load_eval_prompts` | `ValueError: Missing expected_vulnerable ...` | ✅ |
| 单类数据集（全正） | `_assert_dataset_sanity` | `RuntimeError: only one class present (pos=2, neg=0)` | ✅ |
| `expected_vulnerable=0`（int） | `_assert_dataset_sanity` | `RuntimeError: ... 必须是 bool` | ✅ |
| JSON 数组样本缺 `expected_vulnerable` | `_normalize_sample` | `ValueError: Missing expected_vulnerable ...` | ✅ |
| `write_research_splits` 评测行缺标签 | `to_research_record(include_output=False)` | `ValueError: Missing expected_vulnerable ...` | ✅ |
| `write_research_splits` 评测仅一类 | `_assert_eval_rows_labeled` | `RuntimeError: only one class present` | ✅ |

### 5.3 Lint

- `evaluation/prompt_loader.py`、`evaluation/evaluator.py`、`scripts/build_eval_fixed.py`、`scripts/build_dataset.py`、`scripts/migrate_dataset_to_research_schema.py`、`scripts/run_thesis_pipeline.py`、`dataset/research_schema.py`、`configs/default.yaml`、`configs/default_run.yaml`、`configs/default_bandit_only_run.yaml` 均无 linter 错误。

---

## 6. 已知待观察点

- `data/eval_expanded.json` 仍被若干旧路径引用（扁平 schema，已含 `expected_vulnerable`）。当前未改动，但**不再是评测主路径**。未来若要完全弃用，建议再在 `migrate_dataset_to_research_schema.py` 基础上做一次迁移。
- `dataset/synthetic_sql.py` 仍能生成无标签合成样本（供 SFT/DPO demo）。已通过 `scripts/build_dataset.py` 的下游路径隔离——这些样本不再有任何途径进入评测集。

---

## 7. 故障排查

如果 `evaluation/evaluate.py` 启动时异常退出：

1. **`FileNotFoundError: 评测文件不存在: .../eval_fixed.json`**  
   → 运行 `.\.venv\Scripts\python.exe scripts\build_eval_fixed.py` 重建。

2. **`ValueError: Missing expected_vulnerable in evaluation sample (id=...)`**  
   → 检查 `data/generation/eval.json` 或 `data/fix/eval.json` 里对应 id 的样本；或重新跑 `dataset/generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42` 再跑 `scripts/build_eval_fixed.py`。

3. **`RuntimeError: Invalid evaluation dataset: only one class present (pos=..., neg=...)`**  
   → 数据源评测集正负样本严重失衡，需要重采样或重新生成。`generate_expanded_dataset.py` 中 `TARGET_EXPECTED_VULNERABLE_FRACTION=0.5`，一般能保证平衡；若此前魔改过该常量请恢复默认。

4. **未启动就报 `bool(row.get("expected_vulnerable", False))` 相关告警**  
   → 说明仍有代码路径未切换到严格模式；在 `rg "get\(\"expected_vulnerable\"" -g "*.py"` 中排查并替换为 `_require_expected_vulnerable`。
