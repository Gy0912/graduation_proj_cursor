# Changelog — 2026-04-22 对抗训练修复（六次加固）

> 堵住 SFT 训练数据里的核心污染通道：`expected_vulnerable=True` 样本的 `output` 字段曾经**直接写入**真·脆弱 SQL 代码，被 SFT 当作 next-token 目标最小化，等同于"手把手教会模型写 SQL 注入"。本次修复把这些样本**原地**改造成三段式对抗训练样本（`[SECURITY WARNING]` + `[EXPLANATION]` + `[SAFE SOLUTION]`），**不删除**任何样本，label 也保持 `expected_vulnerable=True`，让模型学到「识别不安全指令 → 拒绝不安全实现 → 给出参数化替代」。

---

## 1. 背景与危害（Problem）

### 1.1 bug 的形态

旧版 `dataset/generate_expanded_dataset.py` 的 ambiguous 分支是这样写的：

```python
# 旧逻辑（contaminated）
if ambiguous:
    output = _pick_subtle_output(attack, table, col, rng)   # <-- 真·脆弱 SQL 代码
    expected_vulnerable = True
```

而 `training/sft_preprocess.py` 会把 `output` 直接交给 `tokenize_fn` 做 labels，`CrossEntropyLoss` 把它当作目标序列最小化——**SFT 正在把数据集里一切带 `expected_vulnerable=True` 的脆弱 SQL 代码当成"正确答案"来拟合**。模型越训练，越擅长复现这些脆弱模式。

### 1.2 为什么不能靠"删除这些样本"解决

把 `expected_vulnerable=True` 的样本过滤掉看似干净，但会丢失 50% 的正样本，同时破坏评测集 / 数据生成器 / DPO pair 的 `expected_vulnerable` 正负 1:1 平衡，更重要的是**放弃了模型学"识别不安全指令"的机会**。用户明确要求：

> Do NOT remove these samples. Instead, transform them into adversarial training examples so that the model learns:
> 1. To recognize unsafe instructions
> 2. To refuse insecure implementations
> 3. To provide safe alternatives

### 1.3 与前五次加固的关系

| # | 日期 | 关键词 | 关系 |
|---|------|--------|------|
| 一次 | 2026-04-20 | 评测集 `expected_vulnerable` 强标签 | 独立维度（评测端） |
| 二次 | 2026-04-20 | 样本 `id` 不透明字符串契约 | 独立维度（I/O 契约） |
| 三次 | 2026-04-20 | `eval_fixed.json` 单一写入者 | 独立维度（管线拓扑） |
| 四次 | 2026-04-20 | 缺字段 FAIL FAST | 评测 / 指标 / 训练端共用 |
| 五次 | 2026-04-21 | invalid-extraction 语义 | 评测端指标语义 |
| **六次** | **2026-04-22** | **训练目标反污染** | **训练端目标序列（本 changelog）** |

六次加固与前五次**正交**：即使标签 100% 齐全、指标 100% 严格，只要 `output` 字段里还写着脆弱 SQL，SFT 仍然在污染模型。前五次保护的是**评测信号**，本次保护的是**训练信号**。

---

## 2. 修复范围（What changed）

### 2.1 新增 `dataset/adversarial.py`（single source of truth）

把所有对抗训练相关的 marker / 模板 / 校验逻辑集中到一个独立模块，被 `dataset/` + `training/` + `scripts/` 共同引用，禁止再在别处写死。

| 导出 | 作用 |
|------|------|
| `MARKER_WARNING` / `MARKER_EXPLANATION` / `MARKER_SAFE` | 3 段标题字符串常量 |
| `ADVERSARIAL_MARKERS` | 三个 marker 的只读 tuple，供 sanity check / CLI 引用 |
| `build_secure_response(vulnerable_code, table, column, *, attack, rng)` | 按攻击族（`string_concat` / `fstring` / `format_string` / `fake_sanitization` / `orm_misuse` / `parameterized_query` / `indirect_injection`）合成 3 段式输出；SAFE SOLUTION 在 pymysql / sqlite3 / SQLAlchemy 三套强参数化模板里选一 |
| `extract_safe_solution(output)` | 从 3 段响应里抽出 SAFE SOLUTION 代码块 |
| `assert_adversarial_output_format(output)` | 缺 marker / 空 SAFE SOLUTION 立即 raise |
| `contains_vulnerable_sql_pattern(code)` | 9 条 regex（与 `detection/rule_based.py` 平行 vendoring）——`fstring_sql` / `concat_plus_sql` / `format_sql` / `percent_format_sql` / `percent_format_sql_assigned` / `sqlalchemy_text_fstring` / `sqlalchemy_text_format` / `execute_plus_concat` / `join_build_sql` |
| `DatasetCheckReport` / `check_adversarial_dataset(records)` | 对整份数据集扫描，产出违规明细；函数**本身不 raise**，交由调用方决定 FAIL FAST |

**SAFE SOLUTION 硬契约**（由 `contains_vulnerable_sql_pattern` 强制）：
- 必须使用驱动占位符（`%s` / `?` / `:name`）+ 参数元组 / dict；
- 不含字符串拼接（`"SELECT ..." + x`）；
- 不含 f-string（`f"SELECT ... {x}"`）；
- 不含 `.format()` / `%` SQL 格式化；
- 不含 `sqlalchemy.text(f"...")` / `text("...{x}...")` 这类 ORM 误用。

### 2.2 `dataset/generate_expanded_dataset.py`

- **`build_one_sample()` 的 ambiguous 分支**：不再直接把 `_pick_subtle_output()` 的结果写进 `output`，而是把它作为 `vulnerable_code` 传给 `build_secure_response()`，用返回的 3 段式文本替换 `output`；`expected_vulnerable=True` 保持不变。
- **`_fill_bucket_list()` 的 ambiguous 分支**：同步对齐——数据生成器有两条路径会走 ambiguous，两条都要打补丁。
- **`_decorate_hard_output()` 的调用位置**：挪到非对抗（`expected_vulnerable=False`）分支，避免 `hard`/`very_hard` 的语气修饰污染固定的 3 段式结构。
- **`build_dpo_pairs()`**：`expected_vulnerable=True` 的 `chosen` 改为已经是对抗响应的 `output`，`rejected` 改为"现场合成的脆弱 SQL 片段"；DPO 偏好信号变成"把模型从脆弱 SQL **推离**、向安全拒绝 + 参数化替代**拉近**"。
- **`main()` 收尾**：生成完 train / eval 两份后自动跑 `check_adversarial_dataset`，打印 `[adversarial:train]` / `[adversarial:eval]` 统计（`total` / `adversarial` / `format_compliance` / `safe_solution_clean` / `negatives_clean`），任一比例 < 100% 或有 violation 直接 `sys.exit(1)` 阻止写出污染数据。
- **`_safe_indirect_chain()`**：原本为"间接调用链"分支合成的"多函数分派但全程参数化"的**安全样本**里，SQL 文本仍用 `"SELECT ... WHERE " + pred` 的静态拼接构造；虽然两端都是静态串，运行时零注入风险，但 `contains_vulnerable_sql_pattern` 里的 `concat_plus_sql` 规则会把"`"SELECT ..."` + `"` 这一 token 序列同构为脆弱模式。为了让**所有**写入训练目标的代码都通过机器扫描，改为「整条 SQL 在辅助函数里一次性返回」，两端都不再出现拼接运算符。

### 2.3 `training/sft_preprocess.py`

- 保留全部样本：不再做 "过滤 `expected_vulnerable=True`"、"删除 `[SECURITY WARNING]` 开头的样本" 之类的变换；formatting 逐字符保持。
- 新增三条公开 API：
  - `assert_adversarial_samples_follow_format(records)`：对每条 `expected_vulnerable=True` 样本断言 3 个 marker 全部出现 + SAFE SOLUTION 非空；返回 `{total, adversarial, format_compliant, format_compliance_rate_pct}`。
  - `assert_no_vulnerable_sql_patterns(records)`：对每条样本按类别扫描（adversarial 只扫 SAFE SOLUTION block；safe 扫整段 `output`），违反即 `RuntimeError`。
  - `run_pretraining_sanity_checks(records)`：两者组合 + 审计日志 `[SFT sanity] total_samples=... adversarial_samples=... format_compliance_rate=... safe_solution_clean_rate=... negative_clean_rate=... vulnerable SQL patterns in ANY output = 0 (hard assert passed)`。

### 2.4 `training/train_lora_sft.py` / `training/train_qlora_sft.py`

在 `load_training_dataset(...)` 之后、`split_dataset(...)` 之前插入一行：

```python
run_pretraining_sanity_checks(records)
```

这是训练进程的**唯一**合法入口处；一旦数据集被污染，`RuntimeError` 会在 tokenizer 加载之前就抛出，不会浪费任何 GPU 分钟。

### 2.5 新增 `scripts/check_adversarial_dataset.py`

独立于训练管线的 CLI 验证器，默认扫描 `data/train_expanded.json` / `data/eval_expanded.json` / `data/combined/train.json`，支持 `--input`/`-i` 多次覆盖。

退出码：
- `0`：全部通过
- `1`：任一样本违反契约
- `2`：I/O / 参数错误

可以接入 CI：`python scripts/check_adversarial_dataset.py` 即可做一次全量体检。

### 2.6 归档旧数据

| 归档路径 | 原路径 |
|----------|--------|
| `data/_archive/pre_2026-04-22_contaminated_sft/train_expanded.json` | `data/train_expanded.json` |
| `data/_archive/pre_2026-04-22_contaminated_sft/eval_expanded.json` | `data/eval_expanded.json` |
| `data/_archive/pre_2026-04-22_contaminated_sft/dpo_pairs.json` | `data/dpo_pairs.json` |
| `data/_archive/pre_2026-04-22_contaminated_sft/combined_train.json` | `data/combined/train.json` |
| `data/_archive/pre_2026-04-22_contaminated_sft/combined_eval_fixed.json` | `data/combined/eval_fixed.json` |
| `data/_archive/pre_2026-04-22_contaminated_sft/generation_train.json` / `generation_eval.json` | `data/generation/*` |
| `data/_archive/pre_2026-04-22_contaminated_sft/fix_train.json` / `fix_eval.json` | `data/fix/*` |

附 `data/_archive/pre_2026-04-22_contaminated_sft/README.md` 解释污染形态 + 新老管线对比 + 重现命令；`data/_archive/README.md` 已同步追加该子目录条目。

---

## 3. 强制约束点（Forcing points）

| 关卡 | 位置 | 行为 |
|------|------|------|
| **唯一 marker 源** | `dataset/adversarial.py::ADVERSARIAL_MARKERS` | 3 段字符串常量的唯一真相源；禁止在其他模块重复定义 |
| **生成器 ambiguous 分支** | `dataset/generate_expanded_dataset.py::build_one_sample` / `_fill_bucket_list` | 调用 `build_secure_response(vulnerable_code, table, col, attack=attack, rng=rng)` 替换 `output`；`_decorate_hard_output` 只作用在非对抗分支 |
| **生成器 DPO** | `dataset/generate_expanded_dataset.py::build_dpo_pairs` | `expected_vulnerable=True` → `chosen=adversarial_output`、`rejected=_dispatch_vulnerable(...)`，与前向目标一致 |
| **生成器收尾** | `dataset/generate_expanded_dataset.py::main` | 跑 `check_adversarial_dataset(train/eval)`，任何 violation → `sys.exit(1)` |
| **Safe 样本静态 SQL** | `dataset/generate_expanded_dataset.py::_safe_indirect_chain` | SQL 字符串一次性返回，不再用 `+` 拼接——即便两端都是静态串 |
| **预训练 pre-flight** | `training/sft_preprocess.py::run_pretraining_sanity_checks` | adversarial 样本 marker 齐整 + SAFE SOLUTION 干净 + 全体 `output` 无脆弱模式；违反即 `RuntimeError` |
| **SFT 主入口** | `training/train_lora_sft.py::main` / `training/train_qlora_sft.py::main` | 加载数据后立刻调用 `run_pretraining_sanity_checks(records)`；污染数据绝不进入 trainer |
| **CLI 验证** | `scripts/check_adversarial_dataset.py` | 独立入口，退出码 0/1/2 供 CI/人工使用 |

**强制不变式**：
- 若 `expected_vulnerable=True` 且 `output` 缺任一 marker → 生成器 / 预训练 / CLI 三处都会拒绝；
- 若任何 `output` 里出现 f-string / 字符串拼接 / `.format` / `%` 格式化 SQL / `text()` 包裹 f-string → 同上；
- 任何 `if foo.startswith("[SECURITY WARNING]"): continue` 这种"偷偷过滤 adversarial 样本"的写法都会破坏 `assert_adversarial_samples_follow_format`——保证数据集 + 训练数据**正负样本比**不被偷偷改变。

---

## 4. 三条示例（Transformed samples）

完整的 3 条抽样落在 `logs/samples_adversarial_2026-04-22.log`。这里只摘录 `output`（训练目标），省略 instruction / input_code。

### 4.1 `attack_type=parameterized_query` / `difficulty=easy` / `task_type=fix`

```
[SECURITY WARNING]
The request asks for pseudo-parameterized code (missing parameter tuple, wrong arity, or percent-formatted SQL literal) where the placeholder only looks like a bind; it is still an injection.

[EXPLANATION]
Without a matching parameter tuple or dict the placeholder is never bound, and formatting the SQL literal via %s or .format() before execute() turns the placeholder into an attacker-controllable token. A correct parameterized call keeps the SQL string static and passes parameters whose length matches the placeholders.

[SAFE SOLUTION]
import sqlite3


def query_row(conn: sqlite3.Connection, value: str):
    sql = "SELECT * FROM customers WHERE phone = ?"
    cur = conn.cursor()
    cur.execute(sql, (value,))
    return cur.fetchall()
```

### 4.2 `attack_type=indirect_injection` / `difficulty=hard` / `task_type=fix`

```
[SECURITY WARNING]
The request hides string concatenation behind helper functions and misleading names so the untrusted value still reaches cursor.execute concatenated into the SQL text.

[EXPLANATION]
Multi-function data flow and misleading names do not change the fact that the untrusted value reaches the cursor concatenated into the SQL text. The only robust defense is to make the SQL static at the call site and let the driver or ORM bind the value as a parameter.

[SAFE SOLUTION]
import sqlite3


def query_row(conn: sqlite3.Connection, value: str):
    sql = "SELECT * FROM customers WHERE email = ?"
    cur = conn.cursor()
    cur.execute(sql, (value,))
    return cur.fetchall()
```

### 4.3 `attack_type=orm_misuse` / `difficulty=medium` / `task_type=generation`

```
[SECURITY WARNING]
The request uses sqlalchemy.text() with a user-controlled fragment baked into the raw SQL, which bypasses the ORM's parameter binding and re-introduces SQL injection.

[EXPLANATION]
Wrapping a concatenated string in sqlalchemy.text() binds nothing: the untrusted value was already merged into the raw SQL before SQLAlchemy saw it. The injection-safe form uses a named placeholder inside text() and passes the value via session.execute(stmt, {"v": value}).

[SAFE SOLUTION]
from sqlalchemy import text
from sqlalchemy.orm import Session


def query_row(session: Session, value: str):
    stmt = text("SELECT * FROM orders WHERE password_hash = :v")
    return session.execute(stmt, {"v": value}).fetchall()
```

三条样本都满足：3 段 marker 齐整；SAFE SOLUTION 段里的 SQL 文本是字符串常量；值通过驱动占位符 + 参数元组/字典绑定；全程无 `+` / f-string / `.format` / `%` 格式化。

---

## 5. 数据集统计（Dataset stats）

### 5.1 生成器收尾日志

```
done train=2200 eval=300 vuln_frac_train=0.500 vuln_frac_eval=0.500 \
  train_adversarial=1100/2200 eval_adversarial=150/300
[OK] total_requested=2500 train=2200 -> E:\graduation_proj_1\data\train_expanded.json
[OK] eval=300 -> E:\graduation_proj_1\data\eval_expanded.json
[OK] train_hard_ratio=0.399 eval_hard_ratio=0.553
[OK] expected_vulnerable_frac train=0.500 eval=0.500 (target≈0.5)
[OK] dpo_pairs=2200 -> E:\graduation_proj_1\data\dpo_pairs.json
[OK] research schema -> E:\graduation_proj_1\data\combined\train.json ,
     E:\graduation_proj_1\data\generation , E:\graduation_proj_1\data\fix
[adversarial] markers enforced: ['[SECURITY WARNING]', '[EXPLANATION]', '[SAFE SOLUTION]']
[adversarial:train] total=2200 adversarial=1100 format_compliance=100.00%
                    safe_solution_clean=100.00% negatives_clean=100.00%
[adversarial:eval]  total=300  adversarial=150  format_compliance=100.00%
                    safe_solution_clean=100.00% negatives_clean=100.00%
```

### 5.2 `scripts/check_adversarial_dataset.py` 独立复核

```
[check_adversarial] markers enforced: ['[SECURITY WARNING]', '[EXPLANATION]', '[SAFE SOLUTION]']
[check_adversarial] file: data\train_expanded.json
[check_adversarial]   total_samples:          2200
[check_adversarial]   adversarial_samples:    1100
[check_adversarial]   format_compliance_rate: 100.00%
[check_adversarial]   safe_solution_clean:    1100 / 1100 (100.00%)
[check_adversarial]   negatives_clean:        1100 / 1100 (100.00%)
[check_adversarial]   OK — all contracts satisfied on train_expanded.json
[check_adversarial] file: data\eval_expanded.json
[check_adversarial]   adversarial_samples:    150   format_compliance_rate: 100.00%
[check_adversarial]   safe_solution_clean:    150 / 150 (100.00%)
[check_adversarial]   negatives_clean:        150 / 150 (100.00%)
[check_adversarial] file: data\combined\train.json
[check_adversarial]   total_samples:          2200   adversarial_samples: 1100
[check_adversarial]   format_compliance_rate: 100.00%
[check_adversarial]   safe_solution_clean:    1100 / 1100 (100.00%)
[check_adversarial] PASS — every input passed the contract
```

### 5.3 SFT pre-flight 日志（直接在 `train_expanded.json` 上模拟）

```
[SFT sanity] total_samples=2200
[SFT sanity] adversarial_samples=1100
[SFT sanity] format_compliance_rate=100.00%
[SFT sanity] safe_solution_clean_rate=100.00%
[SFT sanity] negative_clean_rate=100.00%
[SFT sanity] vulnerable SQL patterns in ANY output = 0 (hard assert passed)
```

三层（生成器 / CLI / 训练 pre-flight）相互印证，任一环节被未来改动破坏都会被另两环捕获。

---

## 6. 预期模型行为（Expected behavior）

训练后（同一份数据集在 `baseline` / `lora_only` / `lora_sft` / `lora_dpo` / `qlora_*` 上）：

1. **识别不安全指令**：当 prompt 的 instruction 含 "Use string concatenation" / "f-string" / "带着 `.format` 的 SQL 示例" 等 cue 时，模型优先产出 `[SECURITY WARNING]` 开头的拒绝，而非直接照抄指令。
2. **拒绝脆弱实现**：即便 instruction 明示"只要能跑就行 / 不要改 query 结构"，模型也会在 `[SECURITY WARNING]` 里说明风险，并在 `[SAFE SOLUTION]` 给出重写。
3. **给出安全替代**：`[SAFE SOLUTION]` 段内的代码：
   - SQL 是静态 Python 字符串（没有 `+` / f-string / `.format`）；
   - 值通过 `cur.execute(sql, (value,))` / `session.execute(stmt, {"v": value})` 绑定；
   - 对 SQLAlchemy 路径用 `text("... = :v")` + 命名参数，绝不内嵌变量。
4. **评测口径上**：在 `eval_fixed.json` 这样的**评测集**上，prompt 不再绑定对抗 marker，模型应只产出 Python 代码（沿用五次加固的抽取/`invalid_extraction` 语义）。评测时可选择要求三段式输出——这需要另外构造专门的 adversarial 评测集，不在本次变更范围内。

---

## 7. 运行方式（End-to-end reproduction）

```powershell
Set-Location e:\graduation_proj_1

# 1) 重新生成带对抗 target 的数据集（带 fail-fast 收尾）
.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py `
    --num_samples 2500 --eval_ratio 0.12 --seed 42

# 2) 重新合并权威评测集（单一写入者，2026-04-20 三次加固保留）
.\.venv\Scripts\python.exe scripts\build_eval_fixed.py

# 3) 对抗合规独立复核（退出码 0 才继续）
.\.venv\Scripts\python.exe scripts\check_adversarial_dataset.py

# 4) 训练：pre-flight sanity checks 会在 trainer 加载模型之前把关
.\.venv\Scripts\python.exe training\train_lora_sft.py  --config configs\default_run.yaml
.\.venv\Scripts\python.exe training\train_qlora_sft.py --config configs\default_run.yaml

# 5) 评测（沿用五次加固的 invalid-extraction 语义）
.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model lora_sft
```

---

## 8. 影响面 & 破坏性变更（Impact）

### 8.1 训练侧（不兼容旧数据）

- 旧 `data/train_expanded.json` / `data/combined/train.json` 等归档到 `data/_archive/pre_2026-04-22_contaminated_sft/`，不再被训练脚本读取；**必须**重新生成。
- 用旧数据训练出的 checkpoint（`outputs/models/lora_sft` / `outputs/models/qlora_sft` / `outputs/models/lora_dpo` / `outputs/models/qlora_dpo`）由于输入分布已改变，**必须**重新训练。

### 8.2 评测侧（无缝）

- `evaluation/evaluate.py` 管线对 `eval_fixed.json` 的契约未变；只要评测集自身通过 `build_eval_fixed.py` 的 pre/post-write 校验，五次加固的指标体系（`valid_only_metrics` / `conservative_metrics` / `strict_metrics` / `extraction_failure_rate`）继续工作。

### 8.3 文档侧（追加）

- `README.md`：新增 §「对抗训练 SFT 反污染（2026-04-22 六次加固）」。
- `PROJECT_STRUCTURE.md`：登记 `dataset/adversarial.py` 与 `scripts/check_adversarial_dataset.py`；`training/sft_preprocess.py` + `training/train_lora_sft.py` + `training/train_qlora_sft.py` 标注「pre-flight sanity checks」。
- `data/_archive/README.md` + `data/_archive/pre_2026-04-22_contaminated_sft/README.md`：登记归档原因与新老管线差异。

---

## 9. 回滚策略（Rollback）

若必须回到 2026-04-22 前的旧行为（**不推荐**，会再次把脆弱 SQL 当作 SFT 目标）：

1. `git revert` 本 changelog 关联的 commit；
2. 把 `data/_archive/pre_2026-04-22_contaminated_sft/` 下的 JSON 还原回 `data/` 对应位置；
3. 在 `training/train_lora_sft.py` / `train_qlora_sft.py` 里注释掉 `run_pretraining_sanity_checks(records)` 的调用。

但此路径会把**训练数据污染**重新引入 SFT / DPO，并让`outputs/*_results.json` 上的所有指标再次失去意义。生产环境请**勿**执行回滚。
