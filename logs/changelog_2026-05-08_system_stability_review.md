# 系统级稳定性审查与修复（2026-05-08）

## A. 根因分析

### P0：会导致 DPO collapse / SFT 效果差

| #        | 问题                                                                                               | 影响                                                                                                                                            | 状态      |
| -------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| **P0-A** | `_safe_fix_from_vulnerable` 硬编码 `SELECT * FROM t WHERE c = ?`——丢弃原始 SQL 的 AND/IN/JOIN 子句 | Fix 任务输出破坏 SQL 语义——`WHERE col=%s AND status=%s` → `WHERE col=%s`（丢弃 AND 子句），`IN (%s,%s)` → `= %s`。模型学到「修复=破坏查询结构」 | 🔧 已修复  |
| **P0-B** | `sft_preprocess.py` tokenization 使用 `add_special_tokens=False`                                   | SFT 训练缺少 BOS token——StarCoder2 的 `<                                                                                                        | endoftext | >` 序列边界缺失，训练-推理分布偏移 | 🔧 已修复 |
| **P0-C** | `dpo_train.py` / `train_qlora_dpo.py` beta 回退值 0.01                                             | 若配置缺失/变更，DPO KL 约束弱 10 倍，灾难性遗忘高风险                                                                                          | 🔧 已修复  |

### P1：会劣化 SFT 质量但不会直接崩溃

| #    | 问题                                                            | 影响                                   | 状态                             |
| ---- | --------------------------------------------------------------- | -------------------------------------- | -------------------------------- |
| P1-A | `indirect_injection` 占 DPO 对 33%——攻击类型严重偏斜            | 模型过度关注间接注入，忽略其他攻击类型 | ⚠️ 已知，来自 ATTACK_WEIGHTS 设计 |
| P1-B | hard 样本仅在代码结构上有差异（avg 1.0 func），无实质性难度区分 | hard/easy 标签的指导价值有限           | ⚠️ 设计限制                       |

### P2：低优先级

| #    | 问题                                                    | 影响                   | 状态     |
| ---- | ------------------------------------------------------- | ---------------------- | -------- |
| P2-A | `sql-injection-dataset/` CSV 文件格式不兼容（非 UTF-8） | 无法直接纳入数据集生成 | ⚠️ 不阻塞 |

## B. 修复内容

### Fix 1 (P0-A)：`_safe_fix_from_vulnerable` SQL 语义保存

**修复前**：
```python
# medium: SELECT * FROM t WHERE col = %s AND status = %s
# 修复生成: SELECT * FROM t WHERE col = %s   ← AND status = %s 丢失！
```

**修复后**：
新增 `_extract_sql_template_from_vuln()` + `_ast_value_to_sql_template()`：
1. AST 追溯 execute() 的 SQL 参数到赋值语句
2. 递归遍历 BinOp(Add) 链 / JoinedStr，字符串常量保留，变量引用→`%s`
3. 生成结构完整的安全 SQL

```
medium: WHERE price = %s AND status = %s → WHERE price = %s AND status = %s ✅
hard:   WHERE price IN (%s, %s)        → WHERE price IN (%s, %s)        ✅
```

**兼容性**：回退路径保留（`_extract_sql_template_from_vuln` 返回 None 时使用旧版 table/col 提取）。

### Fix 2 (P0-B)：SFT tokenization BOS

```
# 修复前
p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
# 修复后
p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
```

与 DPO P0-3 修复一致。

### Fix 3 (P0-C)：beta 回退值

```python
# dpo_train.py + train_qlora_dpo.py
beta = float(dcfg.get("beta", 0.1))  # 曾为 0.01
```

## C. 风险分析

| 问题                          | 评估                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| 是否影响旧 checkpoint         | **P0-A**: 是——旧 fix-task SFT checkpoint 学习了被破坏的 SQL 语义。需重新生成数据并重训 SFT |
| 是否影响旧数据                | **P0-A**: 是——`train_expanded.json` 和 `dpo_pairs.json` 需 regenerate                      |
| 是否影响 eval                 | 否——eval 读取 `eval_expanded.json`（评测 prompt），不受影响                                |
| 是否影响其他模型              | 否——AST 通用，`add_special_tokens=True` 对所有 tokenizer 标准                              |
| 是否影响 TRL                  | 否——仅改 tokenization 输入侧，数据格式不变                                                 |
| 是否引入新 distribution shift | P0-A 修复使 fix-task 输出与实际输入结构对齐，**消除**现有的错误 distribution shift         |

## D. 验证方案

### 验证 SFT 数据正常

```powershell
python dataset/generate_expanded_dataset.py --num_samples 2500 --seed 42
python -c "
import json; import ast
data = json.load(open('data/train_expanded.json'))
# Check: all fix task outputs have same SQL structure as input
for r in data:
    if r['task_type'] != 'fix': continue
    inp_placeholders = r['input'].count('%s') + r['input'].count(':v') + r['input'].count('?')
    out_placeholders = r['output'].count('%s') + r['output'].count(':p')
    if inp_placeholders > out_placeholders:
        print(f'WARN: {r[\"attack_type\"]}/{r[\"difficulty\"]} lost columns')
        break
else:
    print('OK: all fix task SQL structures preserved')
"
```

### 验证 DPO pair 正常

```powershell
python dataset/generate_expanded_dataset.py --num_samples 2500 --seed 42
# 检查输出中的行：
# [DPO] structural skip: 0/N  ← 应为 0
# [DPO] pairs generated: N (adversarial only); identical skipped: 0
```

### 验证训练不会再次 collapse

1. 前 50 step 监控：`grad_norm < 5`
2. `logps/chosen > -500`（旧版 collapse 时降至 -800 以下）
3. 无 NaN logits
4. 第 60 step 不崩溃（旧版 collapse 点）

### DPO pair sanity check

```python
from dataset.generate_expanded_dataset import _validate_dpo_pair_structure
import json

with open("data/dpo_pairs.json") as f:
    dpo = [json.loads(l) for l in f if l.strip()]

for r in dpo[:10]:
    v, reason = _validate_dpo_pair_structure(r["chosen"], r["rejected"], {})
    print(f"  {r['attack_type']}/{r['difficulty']}: {'OK' if v else reason}")
```
