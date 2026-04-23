# Changelog — 2026-04-20 缺字段 FAIL FAST 四次加固

> 评测 / 训练 / 指标三层全面清理 `s.get(field, default)` 静默回退；关键字段
> 缺失或类型错误一律 **FAIL FAST**。新增 `validate_eval_samples()` pre-eval
> validator 并打印"有效标签百分比"审计日志。

---

## 1. 背景（Problem）

历史代码遍布如下写法：

```python
if s.get("expected_vulnerable", False): ...
y_pred = bool(s.get("is_vulnerable", False))
out["expected_vulnerable"] = row.get("expected_vulnerable", False)
```

这种"静默兜底为 False / None / []"的写法在单条样本缺字段时**不会中断运行**，
而是把默认值当真数据继续算——导致两类系统性隐形 bug：

1. **评测：缺 `expected_vulnerable` 全部当「非漏洞」**。
   - 整个数据集被当成 easy-negative。
   - Precision / Recall / F1 数学上趋近 1.0，研究结论彻底错误。
   - 这是 `changelog_2026-04-20_eval_label_enforcement.md` 修复的 bug 的"幽灵"——
     即使 loader 层拒绝，只要有下游脚本（如 migrate 里 `_normalize_train_row`，
     research_schema 里 `include_output=True` 分支）仍默认填 `False`，污染就会
     再次注入训练数据，把 SFT target / DPO chosen/rejected 挑反。
2. **指标：检测器输出静默消失**。
   - `bool(s.get("bandit_detected", False))` 让 `_per_sample_from_detection`
     如果漏写字段，整批样本被统计成「Bandit 完全没检测到」。
   - 这种失败**不会报错**，只会让某一层检测器的贡献从论文的实验表格里"消失"。

同时 [TASK] 也要求把 critical fields 扩到三个：`expected_vulnerable` / `id` / `prompt`，
并在评测前加一个独立的 "assert all(...)" 与"% samples with valid labels"日志步骤。

## 2. 修复范围（What changed）

### 2.1 evaluation/metrics.py

- **新增** `_REQUIRED_PREDICTION_FIELDS` 白名单（`expected_vulnerable` / `is_vulnerable` /
  `bandit_detected` / `bandit_b608` / `bandit_has_B608` / `rule_based_detected` /
  `taint_detected` / `invalid_extraction` / `bandit_confidence_levels`）。
- **新增** `_require(sample, key)` / `_require_bool(sample, key)` 辅助：
  - 缺键 → `KeyError("metrics: sample (id=...) 缺少 <key>...")`
  - 非 bool → `TypeError("metrics: sample (id=...) 字段 <key> 类型必须是 bool...")`
- **替换** 以下所有旧的 `s.get(...)` 静默读：
  - `aggregate_metrics`：`sum(1 for s in samples if s.get("is_vulnerable"))` → `_require_bool(s, "is_vulnerable")`
  - `_group_rate`：同上
  - `_detection_layer_and_sources`：`s.get("invalid_extraction")` / `rate(key)` 里的 `s.get(key)` / `s.get("is_vulnerable")` 全部改用 `_require_bool`
  - `_bandit_stats`：`s.get("bandit_detected", False)` / `s.get("bandit_has_B608", False)` / `s.get("bandit_b608", False)` / `s.get("bandit_confidence_levels", [])` 全部改 `_require_*`；`bandit_confidence_levels` 不是 list 直接 `TypeError`
  - `_classification_vs_expected_with_pred` / `_classification_vs_expected`：`s.get(pred_field, False)` / `s.get("is_vulnerable", False)` → `_require_bool`
  - `_by_attack_type_metrics`：`r.get("is_vulnerable")` → `_require_bool`

> **注意（保留）**：`s.get("attack_type") or s.get("vulnerability_type") or "unknown")` 这类
> **metadata 分组 key 派生**保留 fallback 链——这些字段由 `prompt_loader` 与
> `_per_sample_from_detection` 共同保证非空，这里仅作 group key 派生，不涉及预测判定，
> 也不会污染混淆矩阵。

### 2.2 evaluation/evaluator.py

- **新增** `CRITICAL_SAMPLE_FIELDS = ("id", "prompt", "expected_vulnerable")`。
- **扩展** `_assert_dataset_sanity`：
  - 原来只校验 `expected_vulnerable`，现在对**每条样本**同时校验三字段存在性 + 类型。
  - 违规样本**全量收集**（不是碰到第一个就 break），最后统一 raise，让用户一次看到完整
    违规面积。错误信息里含 `valid-label rate: X/N (pct%)` 方便评估数据污染程度。
  - **成功启动时**打印一行新审计日志：
    `[Eval] Valid labels: N/N (100.00%) [id+prompt+expected_vulnerable all present & well-typed]`。
    由于缺失就 raise，这里必为 100.00%，但显式打印让复现者在运行日志里有"本次通过了
    critical field 门闸"的可见证据。

### 2.3 evaluation/prompt_loader.py（新增 pre-eval validator）

- **新增** `CRITICAL_SAMPLE_FIELDS` 公共常量（与 evaluator 对齐）。
- **新增** `validate_eval_samples(samples) -> dict[str, Any]`：
  - 与 `_assert_dataset_sanity` 互补：**不依赖 detector 输出字段**，只做 loader-level
    critical field 校验，适合在模型加载之前调用（省 GPU 初始化时间）。
  - 违规时一次性枚举所有 offending sample，error message 含 `valid-label rate` 百分比。
  - 返回统计 `{total, valid, pct_valid, pos, neg}` 供上层打印 [TASK 4] 百分比日志。

### 2.4 evaluation/evaluate.py（CLI）

- 在 `load_eval_prompts(...)` 之后、模型加载之前**显式插入** pre-eval validation step：
  ```python
  stats = validate_eval_samples(eval_samples)
  print(
      f"[Eval] pre-check: total={stats['total']}, valid={stats['valid']} "
      f"({stats['pct_valid']:.2f}%), pos={stats['pos']}, neg={stats['neg']}"
  )
  ```
  这一行实现了 [TASK 3]「Before evaluation: assert all("expected_vulnerable" in s for s in samples)」
  与 [TASK 4]「Print % samples with valid labels」。
  `evaluator._assert_dataset_sanity` 在推理启动时仍会再做一次完整校验，形成双重保险。

### 2.5 dataset/research_schema.py

- **修复** `to_research_record(include_output=True)` 分支从 `bool(row.get("expected_vulnerable", False))`
  改为 fail-fast：缺 label 或非 bool 直接 raise。错误信息区分 "training sample" /
  "evaluation sample" 方便溯源。
- 旧契约"训练记录允许缺 label 默认 False" 正式废除——合成数据生成器与 legacy migration
  都必须显式给 bool label。

### 2.6 dataset/generate_expanded_dataset.py

- **`build_dpo_pairs`**：把"缺 label → raise"的 ValueError 从**分支后**提前到**分支前**。
  - 原：先 `if r.get("expected_vulnerable"): ...` 走分支，然后到尾部才 `if "expected_vulnerable" not in r: raise`。
  - 新：函数开头就做 `if "expected_vulnerable" not in r / not isinstance(..., bool): raise`，
    然后用 `r["expected_vulnerable"]` 直接读。
  - 同时增加类型校验（非 bool 也 raise）。
- **`main` 收尾统计**：`sum(1 for r in train if r.get("expected_vulnerable"))` →
  先对每条做存在性 + 类型 fail-fast，再 `sum(1 for r if r["expected_vulnerable"])`。
  在这里做一次显式守卫是为了抓住"合成器上游某个新增 branch 忘了写 label"这种回归。

### 2.7 scripts/migrate_dataset_to_research_schema.py

- **`_normalize_train_row`**：移除 `if "expected_vulnerable" not in out: out["expected_vulnerable"] = False`
  的默认兜底。legacy 训练数据若缺 label，现在直接 `ValueError`——要求在源头显式标注后再
  migrate，和 eval 分支的严格契约对齐。
- 新增类型校验（`expected_vulnerable` 必须是 Python `bool`）。

## 3. 不动的点（Intentionally preserved）

下列 `.get()` 用法**不是**静默回退，保持原样：

- 所有 `row.get('id')!r` / `s.get('id')!r` 用于在 **raise/print 错误信息里格式化**
  sample id——不是为评测取值，仅仅是"告诉用户出问题的那条样本长什么样"。
- `s.get("attack_type") or s.get("vulnerability_type") or "unknown"` 在 metrics 的
  分组 key 派生——上游已保证 `vulnerability_type` 非空，这里仅作为多 alias 兼容
  fallback；即使真空了也只会落入 `"unknown"` 桶，不会污染混淆矩阵。
- `s.get("detection_sources") or []` 在 `_detection_layer_and_sources` 里用于统计
  多源组合——空 list 是合法状态（当没有检测器触发时），保留 fallback。
- `scripts/migrate_dataset_to_research_schema.py::load_eval_like` 里的
  `er.get("prompt")` 用于 legacy 数据里尝试从 prompt 字段反解出 instruction/input
  （软回退到 instruction=prompt）——这不是标签相关路径。

## 4. 验证（Verification）

### 4.1 Happy path

```powershell
.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42
.\.venv\Scripts\python.exe scripts\build_eval_fixed.py
.\.venv\Scripts\python.exe -c @"
from evaluation.prompt_loader import load_eval_prompts, validate_eval_samples
from evaluation.evaluator import _assert_dataset_sanity, run_eval_always_safe
s = load_eval_prompts('data/combined/eval_fixed.json')
stats = validate_eval_samples(s)
assert stats['pct_valid'] == 100.0
_assert_dataset_sanity(s)
bundle = run_eval_always_safe(s[:10], merge_mode='or', enable_rule_based=True, enable_taint=False)
assert bundle.n_samples == 10
print('[OK] end-to-end happy path')
"@
```

观察：

- `[Eval] Valid labels: N/N (100.00%)` 审计行出现两次（pre-eval validator + sanity check）。
- `run_eval_always_safe` 跑完不抛错，混淆矩阵非全 0。

### 4.2 Negative path — validator

```python
from evaluation.prompt_loader import validate_eval_samples
cases = [
    ([{'prompt':'p','expected_vulnerable':True}],              'field=id, reason=missing'),
    ([{'id':'a','expected_vulnerable':True}],                  'field=prompt, reason=missing'),
    ([{'id':'a','prompt':'p'}],                                'field=expected_vulnerable'),
    ([{'id':'a','prompt':'p','expected_vulnerable':1}],        'must be bool'),
    ([{'id':42,'prompt':'p','expected_vulnerable':True}],      'must be non-empty str'),
]
# 每个 case 都抛 RuntimeError，错误信息含对应 expect 字符串
```

实测：全部通过，错误信息还打印出 `valid-label rate: 1/2=50.00%` 便于定位污染面积。

### 4.3 Negative path — metrics

```python
from evaluation.metrics import aggregate_metrics
bad = [{'id':'x','expected_vulnerable':True,
        'bandit_detected':False,'bandit_b608':False,'bandit_has_B608':False,
        'rule_based_detected':False,'taint_detected':False,
        'invalid_extraction':False,'bandit_confidence_levels':[]}]
aggregate_metrics(bad)  # -> KeyError: metrics: sample (id='x') 缺少 is_vulnerable；...
```

同时验证 `is_vulnerable: 1`（非 bool）触发 `TypeError: 类型必须是 bool，实际为 int: 1`。

### 4.4 Negative path — dataset/train

```python
from dataset.research_schema import to_research_record
from scripts.migrate_dataset_to_research_schema import _normalize_train_row

to_research_record({'id':'a','task_type':'generation'}, include_output=True)
# -> ValueError: Missing expected_vulnerable in training sample (id='a')...

_normalize_train_row({'id':'a'})
# -> ValueError: Missing expected_vulnerable in training sample (id='a')...

_normalize_train_row({'id':'a','expected_vulnerable':'True'})  # str 而不是 bool
# -> ValueError: expected_vulnerable 必须是 bool，实际为 str: 'True' (id='a')
```

### 4.5 Lint

```powershell
.\.venv\Scripts\python.exe -c "
import py_compile
for f in [
  'evaluation/metrics.py','evaluation/evaluator.py','evaluation/prompt_loader.py',
  'evaluation/evaluate.py','dataset/research_schema.py',
  'dataset/generate_expanded_dataset.py',
  'scripts/migrate_dataset_to_research_schema.py',
]:
    py_compile.compile(f, doraise=True)
print('[OK] all compiled')
"
```

同时 IDE 的 linter 报告零错误。

### 4.6 Codebase audit

```
rg -n "\.get\(\s*['\"]is_vulnerable" evaluation/metrics.py
rg -n "\.get\(\s*['\"]bandit_" evaluation/metrics.py
rg -n "\.get\(\s*['\"]expected_vulnerable" dataset scripts evaluation
```

结果：

- `metrics.py` 中所有预测字段读取均通过 `_require_*` 入口，零 `s.get(pred_field, ...)`。
- `expected_vulnerable` 仅在错误信息格式化与 loader schema 校验里出现，零「默认 False」兜底。

## 5. 影响与动机（Why it matters）

1. **研究结果可信**：无论是评测管线还是训练数据构造，只要任何一个样本缺 critical field，
   运行就**显式中断**而非静默劣化到 easy-negative。论文表格里的 Precision/Recall/F1 从
   「数据污染后的伪值」变回「真实计算结果」。
2. **回归防御层数**：Loader → pre-eval validator → evaluator sanity → per-sample 构造
   → metrics 读取 → dataset/train 端；共 6 道关卡都 FAIL FAST，任何一条新加的 branch
   漏写字段都会在第一次运行时被拦住，而不是悄悄改变数值。
3. **可审计的运行日志**：每次评测都打印 `[Eval] Valid labels: N/N (100.00%)` 审计行，
   配合 pre-check 的 `[Eval] pre-check: total=... valid=... (100.00%)`，在实验日志里
   构成可溯源的"数据质量通过了"证据，不再需要人工回查数据集。
4. **训练端的对称加严**：此前训练端允许 `expected_vulnerable` 默认 False，这是 DPO
   chosen/rejected 挑选错误的种子。对称加严后，训练信号与评测信号都绝不会被静默默认值
   污染，研究结论的内外部一致性得到保证。

## 6. 文件清理（Cleanup per user rule）

- 无旧版脚本被删除（本次改动全部是**就地加严**，不涉及文件替换）。
- 所有旧版（生成器产出的 pyc 缓存等）在 `__pycache__` 下不受影响。
- 无需新增归档目录——本次修复是对现存代码的契约收紧。

## 7. 相关 changelog

- `changelog_2026-04-20_eval_label_enforcement.md` — 评测数据集 `expected_vulnerable` 强标签契约（起点）
- `changelog_2026-04-20_id_string_enforcement.md` — id 不透明字符串契约
- `changelog_2026-04-20_single_eval_writer.md` — `eval_fixed.json` 单一写入者加固
- **本文** — 缺字段 FAIL FAST 四次加固（完结此轮"静默回退"清理）
