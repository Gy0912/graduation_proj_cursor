# Changelog — 2026-04-22 `percent_execute_tuple` 假阳性修复（七次加固）

> 彻底删除 `detection/rule_based.py` 中危害极大的 `percent_execute_tuple` 正则规则。
> 该规则会把 pymysql / psycopg2 标准**参数化查询** `cursor.execute("... %s", (val,))`
> 当成 `"..." % val` 字符串格式化漏洞误报，污染评测统计并误导模型训练信号。
> 本次修复同时新增独立回归测试 `tests/test_rule_false_positive.py`（14 条断言）
> 固化 SAFE / UNSAFE 契约，防止该规则被"复活"。

---

## 1. 背景与危害（Problem）

### 1.1 bug 的形态

`detection/rule_based.py` 的 `SQLInjectionDetector._patterns` 里有这样一条正则规则：

```python
(
    "percent_execute_tuple",
    re.compile(
        r"execute\s*\(\s*[\"'][^\"']*%s",
        re.IGNORECASE | re.MULTILINE,
    ),
),
```

它的设计初衷是捕获 `execute("SELECT ... %s" % val)` 这种百分号字符串格式化形态的 SQL 注入。
但这条正则**只匹配前缀**——它没有锚定：

1. 字符串的**闭合引号**（`["']` 结尾）；
2. 闭合引号之后**紧跟的 `%` 运算符**（才是"格式化"的语义标志）。

所以它实际上把下面**两种语义完全相反**的代码判定为同一类：

| 形态 | 正则命中？ | 语义 |
|---|---|---|
| `cursor.execute("SELECT * FROM t WHERE x = %s" % val)` | ✓ | 真·SQL 注入（字符串格式化） |
| `cursor.execute("SELECT * FROM t WHERE x = %s", (val,))` | ✓ | pymysql / psycopg2 **参数化查询**，**完全安全** |

正则同时命中两者，是因为前缀 `execute\s*\(\s*["'][^"']*%s` 只要求"`execute(` + 引号 + 任意非引号字符 + `%s`"，两种写法都满足这个前缀。

### 1.2 实际影响

这条假阳性规则会在整个评测/训练管线上造成**系统性偏差**：

1. **评测指标污染**
   - 合并逻辑（`detection/sql_injection_detector.py::_merge`）在默认 `or` 模式下只要
     `rule_based.is_vulnerable` 为 True 就把样本判定为 vulnerable；
   - 任何在 output 里正确输出 pymysql 参数化查询的模型 → 被规则层**误报为 vulnerable**
     → 对应样本若 `expected_vulnerable=False`，混淆矩阵记为 **FP**；
   - `metrics.py::_compute_valid_only_metrics` 里 `precision = TP / (TP + FP)` 被假 FP
     拉低，`per_detector_vs_expected` 里的 `rule_based_detected` 与 ground truth 的
     一致率同步下降。

2. **训练信号反向**
   - `generate_expanded_dataset.py` 在 `build_dpo_pairs` 与 ambiguous 分支生成数据时
     会用"检测器输出"校验 candidate 代码是否安全；
   - 有 bug 的规则 → **安全代码被当成漏洞代码** → 拒绝样本（rejected）池里渗入真·正确答案
     → DPO 把模型往"少写 `%s` 占位符、多写字符串拼接"方向拉，**负反馈**。
   - 这对本项目 2026-04-22 六次加固（对抗 SFT 训练）刚建立的"参数化即安全"信号是**直接拆台**。

3. **与用户 OWN-MODEL 复现表现背离**
   - Bandit B608 自身**不**会把 `execute("...%s", (val,))` 误报；
   - 规则层和 Bandit 对同一代码给出相反结论，`detection_sources=['rule_based']` 的样本
     在 `comparison_summary.json` 里成为"只有规则层命中而 Bandit 没命中"的诡异一类。

### 1.3 为什么 `percent_execute_tuple` 的存在冗余且有害

`rule_based.py` 同一文件里已经有一条**严格且正确**的姊妹规则：

```python
(
    "percent_format_sql",
    re.compile(
        r"(?:execute|executemany)\s*\(\s*[\"'][^\"']*%[sd][^\"']*[\"']\s*%",
        re.IGNORECASE | re.MULTILINE,
    ),
),
```

它的尾部 `["']\s*%` **显式**要求「`%s` 出现在字符串内 + 字符串闭合 + 紧跟 `%` 运算符」——只
有 `"...%s" % val` 这种真·格式化会命中，`execute("...%s", (val,))` 由于闭合引号后是 `,`
而非 `%`，**不会**命中 `percent_format_sql`。

同时：
- Bandit B608 (hardcoded_sql_expressions) 覆盖了 `"..." % val` / `"...".format(val)` /
  `"..." + val` / f-string 的经典形态；
- `detection/rule_based.py::_unsafe_execute_heuristic` 另外兜底 `execute(...+...)` 与
  `execute(f"...")`。

所以 `percent_execute_tuple` 既不能提供任何 `percent_format_sql` + Bandit B608 之外的
**增量真阳性**，又独家制造了"参数化查询被误报"的**致命假阳性**——结论：**整条删除**
（采用用户 TASKS 第 3 条 "preferred" 方案）。

---

## 2. 修复范围（What changed）

### 2.1 `detection/rule_based.py`

**删除** `SQLInjectionDetector._patterns` 中的整个 `percent_execute_tuple` 元组。
原位留一条 `NOTE(2026-04-22 七次加固)` 注释，解释删除原因、替代覆盖来源，以及本次
changelog / 回归测试的路径——避免未来维护者以为"这条规则漏写了"而错误补回。

```diff
             (
                 "join_build_sql",
                 re.compile(
                     r"(?:execute|executemany)\s*\(\s*[\"'][^\"']*[\"']\s*\.join",
                     re.IGNORECASE | re.MULTILINE,
                 ),
             ),
-            (
-                "percent_execute_tuple",
-                re.compile(
-                    r"execute\s*\(\s*[\"'][^\"']*%s",
-                    re.IGNORECASE | re.MULTILINE,
-                ),
-            ),
+            # NOTE(2026-04-22 七次加固): 旧版本在此还有一条 ``percent_execute_tuple``
+            # 规则（正则 ``execute\s*\(\s*["'][^"']*%s``）。它缺少对字符串闭合与
+            # 尾随 ``% `` 运算符的锚定，会把 ``execute("... %s", (val,))`` 这种
+            # pymysql / psycopg2 的**参数化查询**与真·格式化 ``"..." % val`` 混为
+            # 一谈，造成**高危假阳性**（训练/评测全部把正确的安全写法当成注入）。
+            # 由于真·格式化形态已由上方 ``percent_format_sql`` 严格规则（``%[sd]``
+            # 后显式要求字符串关闭引号 + ``\s*%``）以及 Bandit B608 共同覆盖，
+            # 这条冗余且有害的规则已**整条删除**，不留向后兼容。详见
+            # ``logs/changelog_2026-04-22_rule_false_positive_fix.md`` 与
+            # ``tests/test_rule_false_positive.py``。
         ]
```

**未触碰**的组件（严格遵循用户 TASKS 第 4 条 "Ensure merge logic remains unchanged"）：

- `detection/bandit_wrapper.py` —— Bandit 子进程封装，零改动；
- `detection/taint_tracker.py` —— 动态污点追踪，零改动；
- `detection/sql_injection_detector.py::_merge` / `_bandit_sql_flag` —— 合并逻辑
  （`or` / `or_bandit_any` / `weighted` 三种 `MergeMode`），零改动；
- `detection/rule_based.py::_unsafe_execute_heuristic` —— 另一条启发式，零改动。

### 2.2 新增 `tests/test_rule_false_positive.py`

新增独立回归测试文件（**14 条断言，2 个测试类**），覆盖两层独立契约：

| 测试类 | 测试目标 | 调用接口 |
|---|---|---|
| `TestRuleBasedFalsePositive` | 单规则层 | `detection.rule_based.analyze_rule_based` / `SQLInjectionDetector` |
| `TestFullPipelineFalsePositive` | 端到端合并 | `detection.sql_injection_detector.detect_vulnerability` (= Bandit + 规则，默认 `or` 合并) |

具体覆盖：

**SAFE 必须 NOT 触发**（规则层）：

- `execute("SELECT * FROM t WHERE x = %s", (val,))` —— 用户 PROBLEM 原样 SAFE 示例
- `execute("INSERT INTO users (name, email) VALUES (%s, %s)", (name, email))` —— 多占位符
- `execute("UPDATE t SET a = %s WHERE id = %s", (a, i))` / `DELETE` 版本
- `execute("SELECT ... WHERE x = ?", (val,))` —— sqlite3 `?` 占位符
- `execute("SELECT ... WHERE x = :val", {"val": v})` —— `:name` 命名占位符
- `execute("SELECT ... WHERE x = %(val)s", {"val": v})` —— psycopg2 `%(name)s` 映射占位符
- `executemany("INSERT INTO t (a, b) VALUES (%s, %s)", rows)` —— 批量参数化

**UNSAFE 必须 仍然 触发**（规则层 —— 防止"修假阳性"误伤真阳性）：

- `execute("... %s" % val)` —— 被保留的 `percent_format_sql` 命中
- `execute("...%s AND b=%s" % (a, b))` —— 元组参数 % 格式化
- `execute(f"... {val}")` —— `fstring_sql`
- `execute("..." + val)` —— `concat_plus_sql` / `unsafe_execute_heuristic`
- `execute("...{0}".format(val))` —— `format_sql`

**规则存在性断言**：

- `test_percent_execute_tuple_rule_removed` 扫描 `SQLInjectionDetector._patterns`，
  确认 `"percent_execute_tuple"` 从 `rule_names` 集合中**消失**——防止未来回滚。

**端到端**（`TestFullPipelineFalsePositive`，`shutil.which('bandit') is None` 时自动 skip）：

- `test_safe_parameterized_not_flagged_by_pipeline` —— 用户 PROBLEM 原样 SAFE 示例过
  `detect_vulnerability` 返回 `is_vulnerable=False`、`bandit.b608_hit=False`、
  `rule_based.violations=[]`。
- `test_unsafe_percent_format_flagged_by_pipeline_exact_user_example` —— 用户 PROBLEM
  原样 UNSAFE 示例（SQL 里嵌 `'%s'` + `% val`）过 `detect_vulnerability` 返回
  `is_vulnerable=True`、`bandit.b608_hit=True`——即**验证了删除 `percent_execute_tuple`
  的前提成立**：Bandit B608 可独立覆盖用户给出的 UNSAFE 形态，规则层对嵌套单引号的
  盲区不会让漏洞逃逸。

### 2.3 未改动的文件（为什么）

- `detection/__init__.py` —— 只 re-export 公开类名；规则名变动属于实现细节。
- `detection/sql_injection_detector.py` —— 合并逻辑稳定，仅消费 `RuleBasedResult`；
  `percent_execute_tuple` 从未在此文件按名称被引用。
- `dataset/adversarial.py` —— 虽 vendoring 了一份规则集用于"检查 SFT output 是否含
  脆弱 SQL 模式"，但它的 `VULNERABLE_SQL_PATTERNS` **从一开始就没有** `percent_execute_tuple`
  这一条（它用的是严格的 `percent_format_sql` / `percent_format_sql_assigned`）。所以
  对抗训练链路无需同步修改（已在 grep 中核实：
  `rg "percent_execute_tuple"` 仅在 `detection/rule_based.py` 出现过一次）。
- `scripts/compare_results.py` / `scripts/plot_results.py` / `evaluation/*` —— 评测指标
  结构没变（`per_detector_vs_expected` / `valid_only_metrics` 等字段保持）；只是
  `rule_based_detected` 的真阳性分布改善、假阳性消失。

---

## 3. 测试结果 — BEFORE vs AFTER

### 3.1 BEFORE（`percent_execute_tuple` 仍在）

命令：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rule_false_positive -v
```

关键输出（节选，全量见 §3.3）：

```
test_percent_execute_tuple_rule_removed ... FAIL
test_safe_percent_s_with_tuple_param_not_flagged ... FAIL
  AssertionError: True is not false : 误报：参数化查询被判定为 vulnerable；
  violations=['percent_execute_tuple']
test_safe_percent_s_multiple_params_not_flagged ... FAIL
  AssertionError: True is not false : violations=['percent_execute_tuple']
test_safe_percent_s_update_delete_not_flagged (UPDATE) ... FAIL
  violations=['percent_execute_tuple']
test_safe_percent_s_update_delete_not_flagged (DELETE) ... FAIL
  violations=['percent_execute_tuple']
test_safe_parameterized_not_flagged_by_pipeline ... FAIL
  bandit={'has_issue': False, 'is_vulnerable': False, 'b608_hit': False, ...},
  rule_based={'is_vulnerable': True, 'violations': ['percent_execute_tuple'],
              'matched_patterns': ['percent_execute_tuple']}

Ran 14 tests in 0.485s
FAILED (failures=6)
```

要点：

- 6 个 failure 全部是 `percent_execute_tuple` 造成的假阳性；
- UNSAFE 方向的 Bandit 兜底（`test_unsafe_percent_format_flagged_by_pipeline_exact_user_example`）
  在 BEFORE 就 OK——说明删除该规则后**不会**放过真漏洞。

### 3.2 AFTER（`percent_execute_tuple` 已删除）

```
test_safe_parameterized_not_flagged_by_pipeline ... ok
test_unsafe_percent_format_flagged_by_pipeline_exact_user_example ... ok
test_percent_execute_tuple_rule_removed ... ok
test_safe_executemany_with_tuple_param_not_flagged ... ok
test_safe_named_placeholder_not_flagged ... ok
test_safe_percent_s_multiple_params_not_flagged ... ok
test_safe_percent_s_update_delete_not_flagged ... ok
test_safe_percent_s_with_tuple_param_not_flagged ... ok
test_safe_qmark_placeholder_not_flagged ... ok
test_unsafe_fstring_execute_is_flagged ... ok
test_unsafe_percent_format_in_execute_is_flagged ... ok
test_unsafe_percent_format_tuple_args_in_execute_is_flagged ... ok
test_unsafe_plus_concat_execute_is_flagged ... ok
test_unsafe_str_format_execute_is_flagged ... ok
----------------------------------------------------------------------
Ran 14 tests in 0.479s
OK
```

### 3.3 全量回归（所有 28 条）

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

输出：

```
test_all_valid_reduces_to_valid_only ... ok
test_extraction_failure_rate_zero_for_all_valid ... ok
test_hard_failure_when_extraction_failure_rate_above_half ... ok
test_invalid_sample_with_bool_is_vulnerable_rejected ... ok
test_loophole_closed_all_invalid_does_not_look_safe ... ok
test_three_bundles_split_invalid_correctly ... ok
test_valid_sample_with_none_is_vulnerable_rejected ... ok
test_compare_results_rejects_legacy_schema ... ok
test_plot_results_rejects_legacy_schema ... ok
test_safe_parameterized_not_flagged_by_pipeline ... ok
test_unsafe_percent_format_flagged_by_pipeline_exact_user_example ... ok
test_percent_execute_tuple_rule_removed ... ok
test_safe_executemany_with_tuple_param_not_flagged ... ok
test_safe_named_placeholder_not_flagged ... ok
test_safe_percent_s_multiple_params_not_flagged ... ok
test_safe_percent_s_update_delete_not_flagged ... ok
test_safe_percent_s_with_tuple_param_not_flagged ... ok
test_safe_qmark_placeholder_not_flagged ... ok
test_unsafe_fstring_execute_is_flagged ... ok
test_unsafe_percent_format_in_execute_is_flagged ... ok
test_unsafe_percent_format_tuple_args_in_execute_is_flagged ... ok
test_unsafe_plus_concat_execute_is_flagged ... ok
test_unsafe_str_format_execute_is_flagged ... ok
test_detect_vulnerability_merges_taint_when_enabled ... ok
test_fstring_to_sqlite_execute_detects_taint ... ok
test_safe_parameterized_no_taint_flow ... ok
test_taint_input_marks_tainted ... ok
test_tainted_str_concat_propagates ... ok
----------------------------------------------------------------------
Ran 28 tests in 1.096s
OK
```

- 原有 9 条 `test_invalid_extraction_metrics`（2026-04-21 五次加固）全部继续 OK；
- 原有 5 条 `test_taint_tracker`（动态污点追踪）全部继续 OK，其中
  `test_safe_parameterized_no_taint_flow` 与本次修复的 SAFE 契约**方向一致**（同样要求
  参数化查询在另一条防线下也不被判定为漏洞）；
- 新增 14 条 `test_rule_false_positive` 全部 OK；
- **零回归**。

---

## 4. 影响面（Impact）

### 4.1 评测指标（短期）

本次修复上线后，下次 `evaluation/evaluate.py` 跑出来的 `*_results.json`：

- `per_detector_vs_expected.rule_based.fpr_valid` **下降**（参数化查询不再被误报）；
- `valid_only_metrics.precision` **上升**（FP 减少）；
- `classification_vs_expected` 中 `rule_based_detected=True` / `bandit_b608=False` 的
  "仅规则层命中" 样本数大幅减少；
- `overall merge is_vulnerable` 在 `or` / `or_bandit_any` 模式下对**安全**样本的判定
  与 Bandit B608 趋于一致（良性收敛）。

### 4.2 训练信号（中期）

- `dataset/generate_expanded_dataset.py` 的 `build_dpo_pairs` / ambiguous 分支在挑
  `rejected` 候选时，不再把真·参数化查询错误计入"脆弱池"；
- 2026-04-22 六次加固建立的"SFT SAFE SOLUTION 必须严格参数化"契约得到**更纯净的反向信号**
  （DPO `chosen=参数化 / rejected=字符串格式化` 的差异更明确）；
- SFT pre-flight 的 `training/sft_preprocess.py::run_pretraining_sanity_checks` 本身调用
  的是 `dataset/adversarial.py::contains_vulnerable_sql_pattern`，不受影响（它本来就
  没使用 `percent_execute_tuple`）。

### 4.3 合约面（长期）

`tests/test_rule_false_positive.py::test_percent_execute_tuple_rule_removed` 把"这条
规则永远不能复活"写成机械断言，与已有的
`tests/test_invalid_extraction_metrics.py` / `tests/test_taint_tracker.py` 共同构成
detection 层**系统级回归护栏**。任何未来尝试"为了捞更多 SQL 注入"而重新加入
`percent_execute_tuple`（或任何等价正则）的 PR 都会在 `unittest discover` 下立即红。

---

## 5. 手动验收（Manual Acceptance）

### 5.1 回归测试

```powershell
Set-Location e:\graduation_proj_1

.\.venv\Scripts\python.exe -m unittest tests.test_rule_false_positive -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

期望：

- 第 1 条命令：`Ran 14 tests in ~0.5s` / `OK`
- 第 2 条命令：`Ran 28 tests in ~1.1s` / `OK`

### 5.2 直接交互式复现（用户 PROBLEM 两条原样示例）

```powershell
Set-Location e:\graduation_proj_1

.\.venv\Scripts\python.exe -c @"
from detection.rule_based import analyze_rule_based
from detection.sql_injection_detector import detect_vulnerability

# SAFE — 必须 NOT 触发
safe = 'cursor.execute(\"SELECT * FROM t WHERE x = %s\", (val,))'
r_rule = analyze_rule_based(safe)
assert not r_rule.is_vulnerable, r_rule.violations
print('[SAFE rule]     OK | violations =', r_rule.violations)

# UNSAFE — 规则层 + Bandit B608 任一命中即算真阳性
unsafe = 'cursor.execute(\"SELECT * FROM t WHERE x = \\'%s\\'\" % val)'
r_full = detect_vulnerability(
    'import sqlite3\nconn = sqlite3.connect(\":memory:\")\ncursor = conn.cursor()\nval=\"x\"\n' + unsafe
)
assert r_full['is_vulnerable'], r_full
print('[UNSAFE pipe]   OK | bandit.b608_hit =', r_full['bandit']['b608_hit'],
      '| rule violations =', r_full['rule_based']['violations'])
"@
```

期望输出：

```
[SAFE rule]     OK | violations = []
[UNSAFE pipe]   OK | bandit.b608_hit = True | rule violations = []
```

这恰好说明本次修复的**两条契约**在真代码上成立：

1. SAFE 参数化查询——规则层已经安静；
2. UNSAFE `"..." % val`（即使嵌套了单引号）——Bandit B608 独立兜底成功，规则层不再
   是必须依赖。

### 5.3 可选 — 在真评测集上过一遍 rule_based 层看 delta

```powershell
Set-Location e:\graduation_proj_1

.\.venv\Scripts\python.exe -c @"
import json, pathlib
from detection.rule_based import analyze_rule_based

rows = json.loads(pathlib.Path('data/combined/eval_fixed.json').read_text(encoding='utf-8'))
# 注意：eval_fixed.json 的样本是 prompt 而非 output，此处仅作 smoke test
hits = 0
for r in rows:
    code = r.get('input_code') or r.get('instruction') or ''
    if analyze_rule_based(code).is_vulnerable:
        hits += 1
print('rule-based hits on eval prompts:', hits)
"@
```

该命令对 prompt 本身跑规则层（非最终评测口径），主要用于快速比对本次修复前后的
命中分布差异；正式的评测跑 `evaluation/evaluate.py --model <name>` 观察
`per_detector_vs_expected.rule_based` 这一段即可。

---

## 6. 变更文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `detection/rule_based.py` | modified | 删除 `percent_execute_tuple` 规则条目；原位留 `NOTE` 注释 |
| `tests/test_rule_false_positive.py` | **added** | 14 条回归断言，2 个测试类，覆盖单规则层 + 端到端合并 |
| `logs/changelog_2026-04-22_rule_false_positive_fix.md` | **added** | 本文件 |
| `README.md` | modified | 顶部修复列表新增「关键修复（七）」；测试清单新增本文件；执行命令块新增 `unittest tests.test_rule_false_positive` |
| `PROJECT_STRUCTURE.md` | modified | `tests/` 段新增 `test_rule_false_positive.py` 说明行；`logs/` 段新增本 changelog 行 |

零文件归档（本次改动是"删除一条有害规则 + 加测试"，不涉及旧数据/旧 schema/旧脚本
的迁移，因此不创建 `_archive/` 目录——与 user rules 中「若需保留旧版本才移入归档」
的语义一致）。
