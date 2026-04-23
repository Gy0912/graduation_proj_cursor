# Changelog — 2026-04-22 对比脚本响应质量指标接入（九次加固）

> 把 `evaluation/evaluator.py::save_results` 自八次加固起就在 `summary.response_quality_metrics`
> 写入的 4 项响应级合规率（`warning_rate` / `explanation_rate` / `safe_solution_rate` /
> `full_compliance_rate`）从 **JSON 里的孤儿字段** 升级为 `scripts/compare_results.py`
> 对比表与 `outputs/comparison_summary.json` 的 **first-class 列**，并新增派生指标
> `structured_response_score = full_compliance_rate` 作为模型排序锚点。**严格不删除 / 不改名**
> 既有的 `sql_injection_rate_valid` / `valid_only` / `conservative` / `strict` /
> `extraction_failure_rate` 字段——本次只做 additive 扩展。新增 14 条独立回归测试，固化
> "缺字段不 crash + 既有指标不变 + 模型间差异可见 + 高质量模型排序更高" 四条边界。

---

## 1. 背景与问题（Problem）

### 1.1 八次加固只走完了一半

2026-04-22 八次加固 (`logs/changelog_2026-04-22_response_quality_metrics.md`) 在
`evaluation/metrics.py` 与 `evaluation/evaluator.py` 两侧把响应级三段式合规率落入了评测
JSON 的 `summary.response_quality_metrics` 块（4 项整体 rate + 8 项正负子集 rate +
`markers` + `note`）。从那一刻起，**单模型** 的评测 JSON 已经能完整表达"响应结构合规度"
这一维度。

但 researcher 在跨模型对比时使用的是 `scripts/compare_results.py` —— 这条脚本里的
`metrics_block_from_eval_json(...)` 仍然 **只抽取**：

```python
return {
    "n_samples": ...,
    "n_valid": ...,
    "n_invalid": ...,
    "extraction_failure_rate": ...,
    "sql_injection_rate_valid": ...,
    "safe_rate_valid": ...,
    "valid_only":   _bundle_metrics(summary, "valid_only_metrics"),
    "conservative": _bundle_metrics(summary, "conservative_metrics"),
    "strict":       _bundle_metrics(summary, "strict_metrics"),
}
```

**没有** `response_quality_metrics`。后果：

| 单模型 JSON 里 | 对比表 / `comparison_summary.json` 里 |
|---|---|
| `warning_rate=0.91` | （不存在） |
| `explanation_rate=0.90` | （不存在） |
| `safe_solution_rate=0.89` | （不存在） |
| `full_compliance_rate=0.87` | （不存在） |

也就是说，researcher 在论文 / slides / 答辩 PPT 里实际能看到的对比表里，**响应质量
指标完全消失** —— 训练侧花了六次加固重塑模型为"对抗式安全教练"的全部努力，在跨模型
对比的 deliverable 里**完全隐身**。任何"训练后 warning/explanation 段被遗忘"的回归在
对比表里**继续无感**，与八次加固前没有本质差别。

### 1.2 用户给出的 GOAL

用户 PROBLEM / GOAL / TASKS 中明确列出：

- `metrics_block_from_eval_json` 必须额外抽取 `warning_rate` / `explanation_rate` /
  `safe_solution_rate` / `full_compliance_rate`；**字段缺失时填 None**，不 crash；
- `_print_table` 必须新增 4 列，以 `0.85 → 85.0%` 的百分比形式显示；
- `comparison_summary` 顶层必须新增 `{method}_full_compliance_rate` /
  `{method}_warning_rate` 等 per-method 键；
- **不准移除** `sql_injection_rate` / F1（valid / conservative / strict）等既有指标；
- 推荐新增派生指标 `structured_response_score = full_compliance_rate` 用于模型排序；
- 验证多模型对比：响应指标在不同模型间应**差异可见**，且高质量模型应**合规率更高**。

---

## 2. 修复范围（What changed）

### 2.1 `scripts/compare_results.py`

| 块 | 变更 | 是否破坏性 |
|---|---|---|
| 顶部 docstring | 增加 §"2026-04-22 九次加固 —— 响应质量指标进入对比表"，列出 4 项变更点 + 兼容契约 + "不删除既有指标" 硬约束 | 否 |
| `RESPONSE_QUALITY_RATE_KEYS` | **新增**模块级常量：4 项整体 rate 的 key 元组（用于 `_response_quality_block` 的统一遍历） | 否 |
| `_optional_float(block, key)` | **新增**辅助：键不存在 / value=None / 类型错都返回 None；专门用于响应质量字段的 opt-in 兼容（旧 evaluator JSON 没有这块时不能 raise） | 否 |
| `_response_quality_block(summary)` | **新增**辅助：从 `summary.response_quality_metrics` 抽 4 项 rate + 派生 `structured_response_score = full_compliance_rate`；缺整块返回 `{key: None for key in 5}` | 否 |
| `metrics_block_from_eval_json` | **扩展**：返回 dict 增加 `"response_quality"` 子键；既有 9 个键 (n_samples / n_valid / n_invalid / extraction_failure_rate / sql_injection_rate_valid / safe_rate_valid / valid_only / conservative / strict) 类型与取值**完全不变** | 否（additive） |
| `_format_pct_cell(value, width)` | **新增**辅助：`0.85 → ' 85.0%'` (width=7)；None → `'    N/A'`（宽度对齐 `f1_strict` / `inj_valid` 等列） | 否 |
| `_print_table` | **扩展** header 与 body 各增加 5 列 `warn% / expl% / safe% / full% / struct%`；既有 8 列（model / n_samples / n_invalid / ext_fail / inj_valid / F1_valid / F1_cons / F1_strict）渲染**完全不变** | 否（additive） |
| `main()` 顶层 summary 写入 | 既有 4 个 baseline_* 键（extraction_failure_rate / sql_injection_rate_valid / safe_rate_valid + sql_injection_reduction_valid_vs_baseline_pct）保留；**新增** 5 个 `baseline_<rate>` + 5 × N 个 `{method}_<rate>` 键 | 否（additive） |
| `main()` 表头打印行 | 文案从 `"=== Summary table (valid-only / conservative / strict) ==="` 改为 `"=== Summary table (valid-only / conservative / strict + response quality) ==="`；新增 `[Legend]` 行解释 5 个新列与 N/A 含义 | 否（仅文案） |

### 2.2 `tests/test_compare_results_response_quality.py` —— 新增

14 条 `unittest` 断言，4 个测试类，覆盖用户 VALIDATION 节的 5 个关键维度：

- **TestMetricsBlockExtractsResponseQuality**（6 条）：
  - `test_full_response_quality_block_extracted` —— 完整块时 4 项整体 rate + struct_score 全部抽出；
  - `test_missing_response_quality_metrics_block_yields_all_none` —— 整块缺失（旧 evaluator JSON）时 5 个键全 None，**不 raise**；
  - `test_partial_missing_keys_yield_per_field_none` —— 单项缺失时该项 None，其他保留实际值；
  - `test_none_value_treated_as_missing` —— 显式 `null` 与缺键等价；
  - `test_non_numeric_value_yields_none_no_crash` —— `"yes"` / `[]` 等非数值被吞为 None，不 ValueError / TypeError；
  - `test_existing_block_keys_unchanged` —— 既有 9 个键（n_samples / valid_only.f1 / conservative.f1 / strict.f1 等）取值与类型完全不变。

- **TestPrintTableFormatsResponseQuality**（3 条）：
  - `test_table_header_includes_all_five_response_quality_columns` —— header 同时包含新 5 列与既有 7 列；
  - `test_percentage_formatting_85_pct_one_decimal` —— `0.85 → 85.0%` 字面量验证（用户原话）；
  - `test_none_renders_as_na_no_crash` —— None 渲染为 `N/A`，既有列不被影响。

- **TestModelsDifferAndRanking**（2 条）：
  - `test_three_models_have_distinct_full_compliance` —— 跨 3 模型 max-min > 0.10 且两两不等；
  - `test_high_quality_model_ranks_higher_by_structured_response_score` —— 按 struct_score 排序，lora_sft (0.85) > lora_only (0.30) > baseline (0.10) 顺序与高质量模型应排前的契约一致。

- **TestSummaryJsonContainsResponseQualityKeys**（3 条）：
  - `test_top_level_summary_has_per_method_response_quality_keys` —— 顶层 summary JSON 出现 5 × N 个 `{method}_<rate>` 键；
  - `test_top_level_summary_keeps_existing_metrics` —— 既有 7 个顶层键（baseline_extraction_failure_rate / 4 × {method}_*）全部保留；
  - `test_legacy_jsons_without_response_quality_dont_crash` —— 1 个 method 是旧 JSON、1 个是新 JSON 时 `main()` 端到端跑通，旧 JSON 顶层值为 null，新 JSON 顶层值正确写入。

### 2.3 `outputs/examples/comparison_summary.example.json`

| 块 | 变更 |
|---|---|
| 顶层 baseline_* 键 | 新增 5 个：`baseline_warning_rate / explanation_rate / safe_solution_rate / full_compliance_rate / structured_response_score` |
| 顶层 `{method}_*` 键 | 新增 5 × 3 = 15 个（lora_only / lora_sft / lora_dpo 各 5 个） |
| `per_model.baseline.response_quality` | 新增子块：5 项 rate（baseline 模型，合规率极低 0.02-0.05） |
| `per_model.lora_sft.response_quality` | 新增子块：5 项 rate（高质量模型，合规率 0.87-0.91） |
| `per_model._legacy_eval_json_example` | **新增伪条目**：演示某模型 JSON 缺 `response_quality_metrics` 时，该 method 的 `response_quality` 子块 5 个键全 null（对比表渲染 N/A、main 不 crash） |
| 既有所有键 | 完全不变 |

### 2.4 `README.md` / `PROJECT_STRUCTURE.md`

详见 §3。

---

## 3. 文档同步

### 3.1 `README.md`

- 顶部"项目概述 + 修复列表"段落增加第 (九) 条引用，链回本 changelog；
- §"汇总对比"块下方插入新子节《对比脚本响应质量指标接入（2026-04-22 九次加固）》，
  包含：(a) 新表格的列含义；(b) `0.85 → 85.0%` 与 N/A 的渲染规则；
  (c) 完整 PowerShell 命令（端到端 setup → 训练 → 评测 → `compare_results.py` → 出图）；
  (d) 期望输出表的真实样本（来自本次修复的 e2e smoke）；(e) 新增 5 个 `{method}_*` 顶层键的 jq 抽取范例；
- §"测试与回归"中：测试总数从 46 → **60**（46 既有 + 14 新增）；新增 `tests/test_compare_results_response_quality.py` 的覆盖维度表。

### 3.2 `PROJECT_STRUCTURE.md`

- `scripts/compare_results.py` 的描述增加 "**2026-04-22 九次加固**：新增 5 列响应质量指标
  并落到顶层 summary 的 `{method}_<rate>` 键……"；
- `tests/` 节新增 `tests/test_compare_results_response_quality.py` 一行（14 条断言、4 个测试类、5 个覆盖维度）；
- `outputs/examples/comparison_summary.example.json` 描述同步增补 "**九次加固**：示例 JSON 增加
  5 项 `baseline_<rate>` + 5 × N 个 `{method}_<rate>` 顶层键，并演示 legacy `_legacy_eval_json_example` 块的全 null 子块"；
- `tests/` 子节里测试总数 46 → 60。

---

## 4. 既有指标不变（"Do NOT remove existing metrics" 硬约束）

机械验证清单（与 `tests/test_compare_results_response_quality.py::TestMetricsBlockExtractsResponseQuality::test_existing_block_keys_unchanged` 一一对应）：

| 既有键 | 类型 | 取值（fixture）| 九次加固后 |
|---|---|---|---|
| `n_samples` | int | 600 | **= 600** |
| `n_valid` | int | 540 | **= 540** |
| `n_invalid` | int | 60 | **= 60** |
| `extraction_failure_rate` | float | 0.10 | **= 0.10** |
| `sql_injection_rate_valid` | float | 0.28 | **= 0.28** |
| `safe_rate_valid` | float | 0.72 | **= 0.72** |
| `valid_only.f1` | float | 0.62 | **= 0.62** |
| `conservative.f1` | float | 0.58 | **= 0.58** |
| `strict.f1` | float | 0.55 | **= 0.55** |

`tests/test_invalid_extraction_metrics.py::TestLegacySchemaRejection::test_compare_results_rejects_legacy_schema`
（旧版 `compare_results.load_summary` 拒绝旧 schema 的契约）**继续通过**——`load_summary`
依然要求 9 个必填字段，旧 JSON 的 `compare_results` 兼容性**没有**因为本次修改而被打开任何
后门（只在 `summary.response_quality_metrics` 这一**新字段**上做软兼容，`extraction_failure_rate`
等 9 个老必填字段缺失依然 ValueError）。

---

## 5. 端到端验证（VALIDATION 节）

在临时目录里合成 7 个 method 的 evaluator JSON，跑一次 `compare_results.main()`，输出对比表：

```
=== Summary table (valid-only / conservative / strict + response quality) ===
model        | n_samples | n_invalid | ext_fail | inj_valid |  F1_valid |  F1_cons | F1_strict |   warn% |   expl% |   safe% |   full% | struct%
-------------+-----------+-----------+----------+-----------+-----------+----------+-----------+---------+---------+---------+---------+--------
baseline     |       600 |        60 |   0.1000 |    0.3400 |    0.7960 |   0.7560 |    0.7260 |   11.0% |    9.0% |    7.0% |    5.0% |    5.0%
lora_only    |       600 |        54 |   0.0900 |    0.3000 |    0.8200 |   0.7800 |    0.7500 |   24.0% |   22.0% |   20.0% |   18.0% |   18.0%
lora_sft     |       600 |        42 |   0.0700 |    0.1600 |    0.9040 |   0.8640 |    0.8340 |   80.0% |   78.0% |   76.0% |   74.0% |   74.0%
lora_dpo     |       600 |        42 |   0.0700 |    0.1300 |    0.9220 |   0.8820 |    0.8520 |   87.0% |   85.0% |   83.0% |   81.0% |   81.0%
qlora_only   |       600 |        60 |   0.1000 |    0.3100 |    0.8140 |   0.7740 |    0.7440 |   22.0% |   20.0% |   18.0% |   16.0% |   16.0%
qlora_sft    |       600 |        48 |   0.0800 |    0.1800 |    0.8920 |   0.8520 |    0.8220 |   77.0% |   75.0% |   73.0% |   71.0% |   71.0%
qlora_dpo    |       600 |        48 |   0.0800 |    0.1400 |    0.9160 |   0.8760 |    0.8460 |   84.0% |   82.0% |   80.0% |   78.0% |   78.0%
[Legend] warn% / expl% / safe% / full% = response_quality_metrics 整体 rate；struct% = structured_response_score (= full_compliance_rate)。缺字段（旧版 evaluator JSON）显示为 N/A。
```

机械验证清单（直接对应用户 VALIDATION 节的 3 条约束）：

| 约束 | 实测结果 |
|---|---|
| **response metrics differ across models** | `full%` 列 7 个值：5.0 / 18.0 / 74.0 / 81.0 / 16.0 / 71.0 / 78.0 —— 全部不同，max-min = 76 个百分点 |
| **values are not all identical** | `warn% / expl% / safe% / full% / struct%` 5 列在 7 个 method 上互不相等（baseline 11→9→7→5→5；lora_dpo 87→85→83→81→81 ……） |
| **high-quality models show higher compliance** | 排序（按 `struct%` 降序）：lora_dpo (81.0%) > qlora_dpo (78.0%) > lora_sft (74.0%) > qlora_sft (71.0%) > lora_only (18.0%) > qlora_only (16.0%) > baseline (5.0%)；DPO ≻ SFT ≻ only ≻ baseline 与训练强度方向**完全一致** |

`outputs/comparison_summary.json` 的 `{method}_*` 顶层键抽样（同次运行）：

```
baseline_full_compliance_rate                              = 0.0500
baseline_structured_response_score                         = 0.0500
lora_only_full_compliance_rate                             = 0.1800
lora_sft_full_compliance_rate                              = 0.7400
lora_dpo_full_compliance_rate                              = 0.8100
lora_dpo_structured_response_score                         = 0.8100
qlora_dpo_full_compliance_rate                             = 0.7800
```

5 项 × 7 method = 35 个新顶层键，全部存在且数值与对比表一致；既有 4 × 7 = 28 个旧
`{method}_*` 键（`extraction_failure_rate` / `sql_injection_rate_valid` / `safe_rate_valid` /
`sql_injection_reduction_valid_vs_baseline_pct`）继续保留。

### 5.1 旧 evaluator JSON 的兼容性验证

把 baseline 的 `*_results.json` 退化成 2026-04-22 八次加固之前的 schema（去掉
`summary.response_quality_metrics`），其他模型保留新 schema。`compare_results.main()`
端到端不 crash：

```
model        | ... |   warn% |   expl% |   safe% |   full% | struct%
-------------+ ... +---------+---------+---------+---------+--------
baseline     | ... |     N/A |     N/A |     N/A |     N/A |     N/A
lora_sft     | ... |   85.0% |   83.0% |   81.0% |   78.0% |   78.0%
```

`comparison_summary.json` 中：

```json
"baseline_warning_rate": null,
"baseline_full_compliance_rate": null,
"baseline_structured_response_score": null,
"lora_sft_warning_rate": 0.85,
"lora_sft_full_compliance_rate": 0.78,
"lora_sft_structured_response_score": 0.78
```

—— 完全符合用户要求："If any field is missing, set to None (do NOT crash)"。

---

## 6. 回归测试（运行结果）

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_compare_results_response_quality -v
```

```
test_existing_block_keys_unchanged ... ok
test_full_response_quality_block_extracted ... ok
test_missing_response_quality_metrics_block_yields_all_none ... ok
test_non_numeric_value_yields_none_no_crash ... ok
test_none_value_treated_as_missing ... ok
test_partial_missing_keys_yield_per_field_none ... ok
test_high_quality_model_ranks_higher_by_structured_response_score ... ok
test_three_models_have_distinct_full_compliance ... ok
test_none_renders_as_na_no_crash ... ok
test_percentage_formatting_85_pct_one_decimal ... ok
test_table_header_includes_all_five_response_quality_columns ... ok
test_legacy_jsons_without_response_quality_dont_crash ... ok
test_top_level_summary_has_per_method_response_quality_keys ... ok
test_top_level_summary_keeps_existing_metrics ... ok
----------------------------------------------------------------------
Ran 14 tests in 0.094s
OK
```

全套回归（46 + 14 = **60**）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```
Ran 60 tests in 0.991s
OK
```

—— 既有 46 条全部继续通过；八次加固在 `metrics.py` / `evaluator.py` 一侧的 18 条
`test_response_quality_metrics` + 9 条 `test_invalid_extraction_metrics`（含
`test_compare_results_rejects_legacy_schema` 这条对 `compare_results.load_summary` 的
拒绝契约）+ 14 条 `test_rule_false_positive` + 5 条 `test_taint_tracker` 全部不受影响。

---

## 7. 影响面总览

| 层 | 文件 | 影响 |
|---|---|---|
| 评测脚本 | `scripts/compare_results.py` | **修改**：扩展 `metrics_block_from_eval_json` / `_print_table` / `main()`；新增 `RESPONSE_QUALITY_RATE_KEYS` / `_optional_float` / `_response_quality_block` / `_format_pct_cell` |
| 测试 | `tests/test_compare_results_response_quality.py` | **新增**：14 条断言，4 个测试类，5 个覆盖维度 |
| 示例 JSON | `outputs/examples/comparison_summary.example.json` | **修改**：增加 5 + 5 × 3 = 20 个新顶层键 + 2 × `response_quality` 子块 + 1 个 `_legacy_eval_json_example` 演示块 |
| 评测 / metrics 核心 | `evaluation/evaluator.py` / `evaluation/metrics.py` | **零改动**（八次加固已经做完，本次只做 downstream 接入） |
| Detection 层 | `detection/*.py` | **零改动** |
| 训练 | `training/*.py` | **零改动**（响应质量指标与训练数据无关，本次更与训练完全解耦） |
| 其他下游脚本 | `scripts/plot_results.py` | **零改动**（用户 TASKS 没要求改它；但本次的兼容契约——`response_quality_metrics` 缺字段不 raise——为以后在 `plot_results.py` 加响应质量柱状图留好了入口） |
| 文档 | `README.md` / `PROJECT_STRUCTURE.md` | **修改**：增加九次加固章节 + 测试总数 46 → 60 + 新 PowerShell 命令 |

---

## 8. 回滚方案

本次修改不涉及数据迁移 / 旧字段删除 / 旧脚本归档，回滚成本极低：

- `git revert <commit>` 一步回滚 `scripts/compare_results.py` + `tests/test_compare_results_response_quality.py` + `outputs/examples/comparison_summary.example.json` + 文档；
- 回滚后既有 46 条测试继续通过，旧版 `comparison_summary.json` 与对比表等价于八次加固
  之前的状态（响应质量指标继续在 evaluator JSON 里"存在但不展示"）。

由于本次只做 additive 扩展、不删字段、不改字段类型，**任何已经写出的 `comparison_summary.json`
文件**（无论是九次加固之前还是之后写的）都可以被新 / 旧版 `compare_results.py` 互读
（旧版只看不到响应质量列，新版会照常显示新列）。

---

## 9. 与既有八次加固的呼应

| 加固 # | 加固范围 | 本次（九次加固）的衔接点 |
|---|---|---|
| 五次（invalid-extraction 语义） | `aggregate_metrics` 拆出 valid_only / conservative / strict 三组指标 | 本次保留这三组 F1 列在表里的位置不变；只在右侧 append 5 个新列 |
| 六次（对抗 SFT 反污染） | 训练侧把 `expected_vulnerable=True` 样本的 output 改成 3 段式 | 本次让对比表能直接看到 6 次加固的"训练成果"在评测端的反馈 |
| 七次（规则层假阳性） | 删除 `percent_execute_tuple` 防止安全代码被误标 | 本次完全独立——规则层修复影响 P/R/F1 列、本次影响响应质量列，互不交叉 |
| 八次（响应质量指标） | 在 metrics / evaluator 两侧落入 `response_quality_metrics` JSON 块 | 本次把这块从"JSON 里的孤儿字段"接到 `compare_results.py` 的对比表与顶层汇总 —— **下游接入**而非新增计算 |

至此，"训练侧三段式契约 → 单模型评测 JSON → 跨模型对比表 → researcher 论文表格"的整条
链路全部打通，没有任何一段把响应质量信息丢掉。

---

## 10. 文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `scripts/compare_results.py` | modified | 扩展 `metrics_block_from_eval_json` / `_print_table` / `main()`；新增 4 个 helper（`RESPONSE_QUALITY_RATE_KEYS` / `_optional_float` / `_response_quality_block` / `_format_pct_cell`）；既有 9 个键的取值与类型完全不变 |
| `tests/test_compare_results_response_quality.py` | **added** | 14 条断言，4 个测试类（`TestMetricsBlockExtractsResponseQuality` 6 + `TestPrintTableFormatsResponseQuality` 3 + `TestModelsDifferAndRanking` 2 + `TestSummaryJsonContainsResponseQualityKeys` 3） |
| `outputs/examples/comparison_summary.example.json` | modified | 顶层新增 5 + 5 × 3 = 20 个 `{method}_<rate>` 键；`per_model.baseline` / `per_model.lora_sft` 各新增 `response_quality` 子块；新增 `per_model._legacy_eval_json_example` 演示 legacy JSON 的全 null 兼容形态 |
| `logs/changelog_2026-04-22_compare_results_response_quality.md` | **added** | 本文 |
| `README.md` | modified | 顶部修复列表 + 测试清单 + 新增《对比脚本响应质量指标接入（2026-04-22 九次加固）》节 |
| `PROJECT_STRUCTURE.md` | modified | `scripts/compare_results.py` / `outputs/examples/comparison_summary.example.json` / `tests/` 节同步九次加固描述；测试总数 46 → 60 |

零文件归档（本次改动是"新增字段读取 + 新增列展示 + 新增测试 + 新增文档"，不涉及旧数据 /
旧 schema / 旧脚本的迁移；既有 `outputs/*_results.json` 与 `outputs/comparison_summary.json`
均不会因 schema 扩展而失效——新增字段缺失会软退化为 None / N/A，不影响任何既有读法）。
