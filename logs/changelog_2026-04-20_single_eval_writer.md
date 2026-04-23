# Changelog — 2026-04-20 · 评测集单一写入者加固（三次加固）

## 一、背景 / 为什么做

在前两轮修复（标签契约、ID 契约）落定后，审计管线时发现 `data/combined/eval_fixed.json` 仍然**同时被两处代码写入**：

| # | 写入路径 | 入口 |
|---|---|---|
| 1 | `dataset/research_schema.py::write_research_splits`（line 145-146 旧版本）| 被 `dataset/generate_expanded_dataset.py:1104` 和 `scripts/migrate_dataset_to_research_schema.py:131` 调用 |
| 2 | `scripts/build_eval_fixed.py::write_eval_fixed` | 用户文档 / 管线 / 配置里声明的「权威合并器」 |

这带来以下**结构性风险**：

1. **Schema 漂移**：两条路径各有自己的 `_assert_eval_rows_labeled` / `_validate_row`，未来任一条被改动都可能让评测集悄悄退化回"缺标签 + 全被当成非漏洞"的破损状态。上一轮的 label enforcement bug 就是这种重复写入+默认 False 回退组合引起的。
2. **去重规则分叉**：`write_research_splits` 走的是 per-task 拆分后直接再把 `eval_r` 写成 `eval_fixed.json`，没有跨 task 的 id 去重 / 统计；`build_eval_fixed.py` 有显式的 `_dedup_key` + `merge_eval_sources` + pos/neg 保证。两套规则共存时，谁最后覆盖谁就看管线顺序，这让行为变得顺序敏感、不可复现。
3. **门闸分散**：`build_eval_fixed.py` 有 schema 强校验（`REQUIRED_KEYS`）和 pos/neg 强制；`write_research_splits` 只有 `_assert_eval_rows_labeled`。如果走前者路径进来的样本缺 `vulnerability_type` / `difficulty`，写出时不会被拦住，直到 evaluation 才炸。
4. **用户约束落地**：用户在本次 [TASK 1] 中明确要求 `scripts/build_eval_fixed.py` 为 canonical builder，并在 [TASK 5] 要求 `run_thesis_pipeline.py` 只调用 canonical builder。现实中 `generate_expanded_dataset` 仍在默默覆盖，违背了声明的契约。

**目标**：让 `data/combined/eval_fixed.json` 只有**唯一合法写入者**（`scripts/build_eval_fixed.py`），所有其它脚本只能产出它的上游输入（per-task 拆分），并在 canonical builder 里把 label 存在性 + schema 一致性 + 正负双类检查做成 pre-write / post-write 双重保险。

## 二、审计结论

完整审计命令：

```powershell
rg "eval_fixed\\.json" --type py -n
rg "write_research_splits" --type py -n
```

结果：

| 位置 | 角色 |
|---|---|
| `scripts/build_eval_fixed.py` | 写入者（保留，唯一） |
| `dataset/research_schema.py::write_research_splits` | 写入者（**移除**本次） |
| `dataset/generate_expanded_dataset.py:1104` | 通过 `write_research_splits` 间接写入（本次失效） |
| `scripts/migrate_dataset_to_research_schema.py:131` | 通过 `write_research_splits` 间接写入（本次失效） |
| `evaluation/*`, `configs/*.yaml`, `scripts/build_dataset.py` docstring, `scripts/run_thesis_pipeline.py` | 仅**读取** / 引用路径名 |

`scripts/build_dataset.py` 自己并不写 `eval_fixed.json`，但 docstring 过时写着 "由 `write_research_splits` 产出"，需要同步更新。

## 三、代码变更

### 1. `dataset/research_schema.py::write_research_splits`

**删除**对 `combined / "eval_fixed.json"` 的写入；**保留** per-task 拆分写入与 `_assert_eval_rows_labeled` 门闸。后者作为 `build_eval_fixed.py` 的上游卫兵依然必需——如果 per-task 拆分自己就缺标签，合并出来的 `eval_fixed.json` 一样会崩。

关键修改：

```33:55:dataset/research_schema.py
def write_research_splits(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    """
    写入研究 schema 数据集：
      - data/combined/train.json  : 全部训练样本（含 output）
      - data/generation/{train,eval}.json / data/fix/{train,eval}.json : 按 task_type 拆分

    **不写 data/combined/eval_fixed.json**。该文件是评测的唯一权威源，必须且仅能
    由 ``scripts/build_eval_fixed.py`` 以 ``data/generation/eval.json`` +
    ``data/fix/eval.json`` 为输入合并产出（单一写入者不变式，由 2026-04-20 加固）。
```

（原 `with open(combined / "eval_fixed.json", "w", ...)` 整段被移除。）

并更新 `_assert_eval_rows_labeled` 的错误文案——它现在守护的是 per-task 拆分，而不是 `eval_fixed.json`。

### 2. `dataset/generate_expanded_dataset.py`

脚本本身只调 `write_research_splits`，行为随之自动失去对 `eval_fixed.json` 的副作用；但末尾 print 仍在指向 `data/combined` 整体，会误导。更新为：

```1122:1128:dataset/generate_expanded_dataset.py
    print(
        f"[OK] research schema -> {ROOT / 'data' / 'combined' / 'train.json'} , "
        f"{ROOT / 'data' / 'generation'} , {ROOT / 'data' / 'fix'}"
    )
    print(
        "[note] data/combined/eval_fixed.json is NOT produced here; "
        "run `scripts/build_eval_fixed.py` to merge generation/eval.json + fix/eval.json."
    )
```

### 3. `scripts/migrate_dataset_to_research_schema.py`

顶部 docstring 与末尾 print 中原先声称自己能产出 `eval_fixed.json`，全部改成仅产出 per-task 拆分；并明确指引下一步需要跑 `build_eval_fixed.py`。

### 4. `scripts/build_eval_fixed.py`

这是本次的主要加强点。

**新增常量**：

```python
SCHEMA_CONSISTENCY_KEYS: tuple[str, ...] = (
    "expected_vulnerable",
    "vulnerability_type",
    "difficulty",
)
```

对应用户 [TASK 4]「All samples must contain: expected_vulnerable / vulnerability_type / difficulty」。

**新增 `_assert_dataset_final(dataset, *, stage)`**：集中承接用户 [TASK 3] 与 [TASK 4]：

1. 空集直接拒绝（`RuntimeError`）；
2. 对每条样本做 `if "expected_vulnerable" not in sample: raise RuntimeError("missing label")`，带 idx + id 定位；
3. 对 `SCHEMA_CONSISTENCY_KEYS` 做完整性扫描，缺任一字段都收集到 `missing_by_idx` 并汇总（前 5 条具体 idx/id/missing，后面 `... and N more`）；
4. 再次核验 `expected_vulnerable` 是严格的 `True`/`False`——避免 JSON 反序列化时 int 1/0 混进来；
5. 最后再跑一次 pos/neg 双类非空检查（与 `merge_eval_sources` 的冗余保险）。

签名里带 `stage`（`pre_write` / `post_write` / `acceptance`），错误信息前缀用它，让故障 trace 一眼能定位是 merge 阶段就脏了还是写盘后脏的。

**新增 `_readback_and_verify(out_path)`**：写入后从磁盘 `json.loads` 重新读回原文件，再对 `data` 跑一次 `_assert_dataset_final(stage="post_write")`。覆盖的风险面：

- JSON 序列化异常（罕见但可能）；
- UTF-8 编码异常 / BOM；
- 外部工具 / 人工编辑在短暂窗口修改了文件；
- 磁盘写坏 / 部分写入。

**修改 `main()`**：调用顺序变成

```python
sources = [args.generation, args.fix]
merged, stats = merge_eval_sources(sources)

_assert_dataset_final(merged, stage="pre_write")   # 写盘前最终校验
write_eval_fixed(merged, args.out)
n_readback = _readback_and_verify(args.out)        # 写盘后读回复核
```

并增加一行 log：

```
[build_eval_fixed] post-write readback verified: 600 samples (expected_vulnerable / vulnerability_type / difficulty all present)
```

### 5. `scripts/run_thesis_pipeline.py`

流水线本身已经按正确顺序调用脚本，无需调整步骤，仅把「单一写入者」不变式写成 module docstring + 每步注释：

```
generate_expanded_dataset.py  -> data/combined/train.json
                                 data/generation/{train,eval}.json
                                 data/fix/{train,eval}.json
build_dataset.py              -> dataset/*.jsonl (SFT/DPO demo only)
build_eval_fixed.py           -> data/combined/eval_fixed.json  [唯一写入者]
```

在三条数据步骤前分别加注释，明确每一步的职责与**不**做的事，防止未来的 contributor 乱挪顺序或新增另一个 eval writer。

### 6. `scripts/build_dataset.py`

docstring 里原文：

> 评测集的唯一权威源是 data/combined/eval_fixed.json，由
> `scripts/build_eval_fixed.py`（或 `dataset/generate_expanded_dataset.py` 中的
> `write_research_splits`）产出

括号里的「或 `write_research_splits`」已不再成立，更新为只提 `build_eval_fixed.py`。

### 7. 文档

- `README.md`：
  - 顶部新增 **关键修复（三）** 横幅，指向本 changelog。
  - 项目结构树更新 `eval_fixed.json` 行（声明单一写入者）、`generate_expanded_dataset.py` 行、`research_schema.py` 行、`build_eval_fixed.py` 行。
  - 「如何运行 → 生成训练集 & 合并权威评测集」说明段重写，突出单一写入者不变式与 pre/post-write 校验。
  - 「执行流程」条目 1-2 重写，把「同时写 `eval_fixed.json`」的错误描述去除。
  - 「评测数据集标签强制约束」表里「源数据合并」/「写出」两行重写。
  - 新增独立章节「**评测集单一写入者（2026-04-20 三次加固）**」，含不变式示意、做这件事的原因、强制约束点表、`手动验收` 段。
  - 「手动验收（回归校验）」的期望输出补入 `post-write readback verified` 行。
  - 底部故障排查指引里加入新 changelog 链接。

- `PROJECT_STRUCTURE.md`：
  - `data/combined/eval_fixed.json` 行更新描述为「**唯一权威评测集；单一写入者 `scripts/build_eval_fixed.py`**」。
  - `generation/` / `fix/` 行描述为 `eval_fixed.json` 的上游**唯一**输入。
  - `dataset/generate_expanded_dataset.py` 与 `research_schema.py` 行更新，强调不写 `eval_fixed.json`。
  - `scripts/` 表格里 `build_eval_fixed.py` / `migrate_dataset_to_research_schema.py` / `run_thesis_pipeline.py` 三行更新。
  - `logs/` 表格新增本 changelog 条目。

## 四、验证

### A. Happy path：`build_eval_fixed.py` 端到端

```powershell
.\.venv\Scripts\python.exe scripts\build_eval_fixed.py
```

结果：

```
[build_eval_fixed] sources: ['E:\\graduation_proj_1\\data\\generation\\eval.json',
                             'E:\\graduation_proj_1\\data\\fix\\eval.json']
[build_eval_fixed] kept per source: {'...\\generation\\eval.json': 300, '...\\fix\\eval.json': 300};
                   dups skipped: {'...': 0, '...': 0}
[build_eval_fixed] total=600 pos=300 neg=300
[build_eval_fixed] wrote E:\graduation_proj_1\data\combined\eval_fixed.json
[build_eval_fixed] post-write readback verified: 600 samples
                   (expected_vulnerable / vulnerability_type / difficulty all present)
```

**关键**：`post-write readback verified` 是本次新加的一行，证明写盘后读回 + 再校验这条路径正常工作。

### B. 结构不变式：`write_research_splits` 不再写 `eval_fixed.json`

跑一个内存中的小样本，验证 `write_research_splits` 只产出 5 个文件，不含 `data/combined/eval_fixed.json`：

```
[OK] produced files:
   data/combined/train.json
   data/fix/eval.json
   data/fix/train.json
   data/generation/eval.json
   data/generation/train.json
[OK] write_research_splits no longer writes eval_fixed.json (invariant held)
```

### C. Negative path：`_assert_dataset_final` 拦截所有异常输入

依次输入 6 种异常数据集，全部被 `RuntimeError` 精确定位拒绝：

| # | 输入 | 捕获的错误信息 |
|---|---|---|
| 1 | 空 `[]` | `Invalid eval dataset: empty dataset (0 samples)` |
| 2 | 第 2 条缺 `expected_vulnerable` | `missing label (idx=1, id='b')` |
| 3 | 第 2 条缺 `vulnerability_type` | `Inconsistent eval schema: 1 samples missing one of [...]. idx=1(id='b', missing=['vulnerability_type'])` |
| 4 | 第 2 条缺 `difficulty` | 同上，missing=['difficulty'] |
| 5 | 全正类 | `only one class present (pos=2, neg=0)` |
| 6 | `expected_vulnerable=1`（int，不是 bool）| `expected_vulnerable must be strictly bool True/False ... (difference = likely int/str leaked in)` |

Minimal valid case（2 条，1 正 1 负，三字段齐整）通过。

### D. Lint

对改动的 5 个文件跑 ReadLints：无任何错误。

### E. 代码级冗余扫描

```powershell
rg "open\([^)]*eval_fixed" --type py -n   # 匹配所有对 eval_fixed.json 的 open 调用
```

仅剩 `scripts/build_eval_fixed.py::write_eval_fixed` 一处命中。✅

```powershell
rg "eval_fixed\.json" --type py -n
```

所有其它命中均在 docstring / print / configs 路径引用里，没有新的隐蔽写入者。

## 五、不变式声明（写给未来的自己 / reviewer）

1. `data/combined/eval_fixed.json` 的**唯一**合法产生方式是运行 `scripts/build_eval_fixed.py`。
2. 任何数据构建/迁移脚本都只能产出它的上游：`data/generation/eval.json` + `data/fix/eval.json`（及对应 `train.json`、`data/combined/train.json`）。
3. `build_eval_fixed.py` 必须保持 pre-write 与 post-write 双重校验——去掉任一条都会让评测集有机会悄悄退化。
4. 当新增评测字段（例如 `cwe_id`、`severity`）时，同时把它加入 `SCHEMA_CONSISTENCY_KEYS`，否则新字段会出现"部分样本有 / 部分没有"的分叉。
5. **禁止** 在任何时候用 `try/except` 包住这些 `RuntimeError` / `ValueError`、或加任何"缺标签则默认 False"类的静默兜底。
