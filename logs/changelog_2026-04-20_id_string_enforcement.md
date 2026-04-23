# Changelog — 2026-04-20 评测样本 ID 契约加固（不透明字符串）

## 摘要

修复一个**关键评测崩溃 Bug**：`evaluation/evaluator.py` 末尾

```python
evaluated_samples.sort(key=lambda x: int(x["id"]))
```

对 `id` 做了**数字假设**。当前评测集 `data/combined/eval_fixed.json` 的 id 已切换为
由 `dataset/research_schema.py::stable_sample_id` 生成的不透明哈希字符串（如
`"sqlsec-fed7e7019058551a9d08"`），`int(...)` 在第一条样本上就会抛：

```
ValueError: invalid literal for int() with base 10: 'sqlsec-fed7e7019058551a9d08'
```

结果：整场评测在最后一步崩溃，本轮生成的全部 per-sample 结果被抛弃、结果 JSON 不写出、
`compare_results.py` 找不到对应文件、可视化脚本空转。即使有 try/except 也会把 600 条
生成结果全部丢掉。

本次更新把 id 的合约从「**可能是字符串或整数**」统一为「**始终是不透明非空字符串**」，
并在四层入口（源合并 / 研究 schema 写出 / 加载 / 评测运行时）各自 FAIL FAST 拒绝非字符串
id，彻底消除数字假设。

---

## 1. 影响范围审计

### 实际触发崩溃的位置

| 文件 | 行 | 原代码 | 问题 |
|------|----|--------|------|
| `evaluation/evaluator.py` | 373（原） | `evaluated_samples.sort(key=lambda x: int(x["id"]))` | 对哈希 id 直接 `int()` → `ValueError`。评测最后一步 100% 崩溃。 |

### 潜在类型污染（类型混用、难定位 bug 温床）

| 文件 | 原代码 | 问题 |
|------|--------|------|
| `evaluation/evaluator.py::_per_sample_from_detection` | `"id": sid if sid is not None else sample_id` | 当 `src["id"]` 缺失时回退到位置索引 int。导致结果 JSON 里 `id` 字段可能是 `str` 也可能是 `int`，下游去重 / 排序行为不可预测。 |
| `evaluation/evaluator.py` invalid_extraction 分支 | `"id": src.get("id", sample_id)` | 同上。 |
| `evaluation/prompt_loader.py::_normalize_sample` | `"id": row.get("id")` | `row` 里没有 id 时 `id=None`，能通过 loader；到评测器才爆。 |
| `scripts/build_eval_fixed.py::_validate_row` | `if not (isinstance(sid, str) and sid.strip()) and not isinstance(sid, int):` | 仍允许 int id，与"id 是字符串"的新契约不一致。 |
| `scripts/build_eval_fixed.py::_dedup_key` | `if isinstance(sid, (str, int)) and str(sid).strip():` | 同上。 |
| `dataset/research_schema.py::to_research_record` | `"id": row.get("id") or stable_sample_id(...)` | 若 `row["id"]` 是一个非空但**非字符串**的值（例如 `123`），会直接穿透 —— 下游收到 int id。 |

### 扫描后确认**不是 bug**（故意保留）

| 位置 | 说明 |
|------|------|
| `evaluation/evaluator.py:312` `sample_id = int(sample_ids[row].item())` | `sample_ids` 是 `torch.arange(len(prompts), dtype=torch.long)`，这里是把 dataloader 的位置索引从 tensor 拆成 int。`sample_id` 变量语义是**位置索引**，不是样本 id 字段。保留原样。 |
| `detection/sql_injection_detector.py:102` `sample_id: int = 0` | 作为 Bandit 临时文件名后缀（`sample_{sample_id}.py`），独立整数计数器，与 dataset id 字段无关。保留原样。 |
| `evaluator.py` 所有 `sample_id` 参数 / `_per_sample_from_detection(sample_id=...)` | 均为位置索引（`sample_index`），非字符串 id 字段。保留原样。 |

---

## 2. 具体修改

### `evaluation/evaluator.py`

1. 新增 `_require_string_id(src) -> str`：严格读取 `src["id"]`，缺失 / 非 str / 空白一律
   `ValueError`。与现有 `_require_expected_vulnerable` 风格一致，形成「样本字段严格读取」
   工具集。
2. `_per_sample_from_detection` 的返回值 `"id"` 改为 `_require_string_id(src)`，
   **移除** `sid if sid is not None else sample_id` 的 int 回退。
3. invalid_extraction 分支的 `"id": src.get("id", sample_id)` 同样改为 `_require_string_id(src)`。
4. `run_eval_on_prompts` 末尾：
   - 旧：`evaluated_samples.sort(key=lambda x: int(x["id"]))`
   - 新：先遍历一次 `evaluated_samples` 做 `isinstance(sample["id"], str)` 断言（用户要求的
     defensive 检查点），再 `evaluated_samples.sort(key=lambda x: x["id"])`。

### `evaluation/prompt_loader.py`

- `_normalize_sample` 开头新增 id 强校验：
  ```python
  sample_id_raw = row.get("id")
  if not isinstance(sample_id_raw, str) or not sample_id_raw.strip():
      raise ValueError(f"id 必须是非空字符串（不透明哈希形式...），实际为 {type(...).__name__}: {...!r}")
  ```
- 返回字典里 `"id": row.get("id")` 改为 `"id": sample_id_raw`（明确使用已校验的字符串值）。

### `scripts/build_eval_fixed.py`

- `_validate_row` 中 id 校验从「字符串 or 整数」收紧为「必须是非空字符串」，错误信息里
  同时打印实际类型与值：
  ```python
  if not isinstance(sid, str) or not sid.strip():
      raise ValueError(f"... id 必须是非空字符串（不透明哈希，如 'sqlsec-<hex>'），"
                       f"实际为 {type(sid).__name__}: {sid!r}")
  ```
- `_dedup_key` 去掉 `int` 分支，非空字符串 id 直接作为 dedup 键（不再走 `str(sid)` 转换）。

### `dataset/research_schema.py`

- `to_research_record` 对 `row.get("id")` 做显式分支：
  - 是非空字符串 → 直接用；
  - 否则（None / int / 空串 / 其它） → `stable_sample_id(row)` 重新生成稳定哈希字符串。
- 效果：该函数**保证**输出 record 的 `id` 字段始终是字符串，自然把任何遗留整数 id 数据
  平滑升级为哈希 id。

### `README.md`

- 顶部「关键修复」块加入第二条（id 契约）指向本次 changelog。
- 新增章节「评测样本 ID 契约（2026-04-20 二次加固）」：
  - 阐述「id 为不透明字符串」假设；
  - 列明四重保险点；
  - 提供正向 / 负向 PowerShell 验收命令。
- 「手动验收」段落加入对 id 字段的非空字符串断言。

### `PROJECT_STRUCTURE.md`

- `logs/` 段落加入新 changelog 条目。

---

## 3. 未修改（刻意保留的决策）

### 不改变 `sample_id` 语义

`evaluation/evaluator.py` 中 `sample_id` 参数 / 局部变量**全部**语义是「dataset 位置索引
（0..N-1）」，不是 id 字段。它会被写入 per-sample 结果里的 `"sample_index"` 键，便于回溯
batch 顺序。本次**不重命名**这些变量，避免触及无关代码并产生噪音 diff。

### 不动 `detection/sql_injection_detector.py::sample_id`

同样是位置索引 / 临时文件名计数器。与 dataset id 字段无关，不在本次修复范围。

### 不使用 try/except 做兜底

按 [IMPORTANT] 明确要求：所有新增校验都是 `raise ValueError(...)`，不做静默回退、不把
`int(id)` 包进 `try/except` 里。错误信息里都带有 `type()` 与 `!r` 打印，便于直接定位源头。

---

## 4. 文件清理

本次修复**不产生**过时代码或冗余数据：

- 修改的四个源文件都是现存文件，原地修复；
- 没有新的数据集文件产生；
- 没有需要归档的遗留文件——历史上产生过整数 id 的 `data/_archive/combined_eval_legacy_unlabeled_2026-04-20.jsonl` 早在 2026-04-20 首次标签契约修复中已被归档到 `data/_archive/`，本次无需再动。

---

## 5. 回归验证（已通过）

### 5.1 端到端冒烟

```powershell
.\.venv\Scripts\python.exe scripts\build_eval_fixed.py
# [build_eval_fixed] total=600 pos=300 neg=300
# [build_eval_fixed] wrote .../data/combined/eval_fixed.json

.\.venv\Scripts\python.exe -c @"
from evaluation.prompt_loader import load_eval_prompts
from evaluation.evaluator import run_eval_always_safe
s = load_eval_prompts('data/combined/eval_fixed.json')
bundle = run_eval_always_safe(s[:10], merge_mode='or', enable_rule_based=True, enable_taint=False)
ids = [r['id'] for r in bundle.per_sample]
assert all(isinstance(x, str) and x.strip() for x in ids)
print('[OK] ids are all non-empty strings')
"@
```

期望 / 实际输出：

```
Detected JSON format
[Eval] Total samples: 10
[Eval] Vulnerable: 5
[Eval] Safe: 5
[OK] ids are all non-empty strings
```

### 5.2 负向路径（都应 `ValueError`，无一静默通过）

| 场景 | 位置 | 期望异常 | 结果 |
|------|------|---------|------|
| `_require_string_id({'id': 123})` | evaluator | `ValueError: ID must be string; got type=int value=123` | ✅ |
| `_require_string_id({'id': None})` | evaluator | `ValueError: ID must be string; got type=NoneType value=None` | ✅ |
| `_require_string_id({})` | evaluator | `ValueError: sample 缺少 id 字段；...` | ✅ |
| `_require_string_id({'id': ''})` | evaluator | `ValueError: ID must be a non-empty string; got ''` | ✅ |
| `_require_string_id({'id': '   '})` | evaluator | `ValueError: ID must be a non-empty string; got '   '` | ✅ |
| `_normalize_sample({'id': 7, ...})` | prompt_loader | `ValueError: id 必须是非空字符串...，实际为 int: 7` | ✅ |
| `build_eval_fixed` 输入 int id | `_validate_row` | `ValueError: ... id 必须是非空字符串...，实际为 int: 0` | ✅ |

### 5.3 排序的确定性

```python
>>> samples = [{'id':'sqlsec-zzz'}, {'id':'sqlsec-aaa'}, {'id':'sqlsec-bbb'}]
>>> samples.sort(key=lambda x: x['id'])
>>> [s['id'] for s in samples]
['sqlsec-aaa', 'sqlsec-bbb', 'sqlsec-zzz']
```

字典序对 `sqlsec-<hex20>` 形式等价于按十六进制哈希的字典序，**确定且稳定**。不同批次运行
只要评测集相同，最终写入 `outputs/*_results.json` 的 `per_sample` 顺序就相同，复现性保持。

### 5.4 Lint

- `evaluation/evaluator.py`、`evaluation/prompt_loader.py`、`scripts/build_eval_fixed.py`、
  `dataset/research_schema.py` 全部 clean，无 linter error。

---

## 6. 对研究/评测的影响

### 立即收益

- **评测不再崩溃**：之前每次跑完 600 条样本要到最后 `sort` 才爆掉，整轮结果被抛弃；
  现在整个管线可以把结果顺利写进 `outputs/*_results.json`。
- **per-sample id 类型稳定**：`outputs/*_results.json::per_sample[].id` 现在保证是
  `"sqlsec-..."` 字符串；`compare_results.py` / `visualization/*` 在聚合时做 id 匹配不再
  受 int-vs-str 类型混淆影响。

### 工程级收益

- **四重保险** 让非字符串 id 无法进入评测：
  1. `build_eval_fixed._validate_row`（源合并）→
  2. `research_schema.to_research_record`（写出时升级）→
  3. `prompt_loader._normalize_sample`（加载时）→
  4. `evaluator._require_string_id` + 排序前扫描（运行时）。
- 任一层错误都会抛带**类型+值**的 `ValueError`，定位直达源头。

### 向后兼容

- **显式不兼容**的契约变更：遗留携带 int id 的外部数据源会在 `build_eval_fixed` / 加载阶段
  直接拒收。这是故意的——旧整数 id 与新哈希 id 混用会污染去重和跨轮比较。
- `dataset/research_schema.py::to_research_record` 对老数据做了**自动升级**（通过
  `stable_sample_id` 重新哈希），所以上游只要跑一次 `generate_expanded_dataset` 或
  `migrate_dataset_to_research_schema` 就能把旧整数 id 平滑替换成哈希 id，不需要手工清洗
  数据。

---

## 7. 故障排查

### 场景 A：跑评测报 `ValueError: ID must be string; got type=int value=...`

说明评测集或 per-sample 结果里混入了整数 id。处理步骤：

1. 确认 `configs/*.yaml` 的 `files.eval_prompts` 指向 `data/combined/eval_fixed.json`（而非
   归档目录的旧文件）。
2. 重新运行 `scripts/build_eval_fixed.py` 重建权威评测集。
3. 如果是第三方 / 手工 JSON，请在源头把 id 改成 `sqlsec-<hex>` 字符串；或把该文件喂给
   `scripts/migrate_dataset_to_research_schema.py` 做 id 升级。

### 场景 B：`ValueError: id 必须是非空字符串...，实际为 NoneType: None`

源 JSON 的某条样本没有 `id` 字段。解决：

- 重新生成：`.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42`
- 或单独升级：`.\.venv\Scripts\python.exe scripts\migrate_dataset_to_research_schema.py --train data\train_expanded.json --eval data\eval_expanded.json`

### 场景 C：`sort` 通过，但 `compare_results.py` 报 id 找不到匹配

排查顺序：

1. `python -c "import json; d=json.load(open('outputs/baseline_results.json')); print({type(r['id']).__name__ for r in d['per_sample']})"` — 应输出 `{'str'}`。
2. 如出现 `{'int'}` 或混合，说明有历史结果文件没重跑；删除对应 `outputs/*_results.json` 并
   重新 `evaluation/evaluate.py --model <name>` 即可。
