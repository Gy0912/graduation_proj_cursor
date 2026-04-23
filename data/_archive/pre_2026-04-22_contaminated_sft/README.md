# data/_archive/pre_2026-04-22_contaminated_sft

本目录保存 2026-04-22 对抗训练修复之前、仍然把脆弱 SQL 写进 SFT ``output`` 的
数据集快照。**仅供审计与回溯**，**任何训练 / 评测脚本都不得再读这些文件**。

## 文件清单

| 归档文件 | 原路径 | 污染形态 |
|----------|--------|----------|
| `train_expanded.json` | `data/train_expanded.json` | 扩展 SFT 训练集：`expected_vulnerable=True` 样本的 `output` 是 `_pick_subtle_output()` 合成的**脆弱 SQL 代码** |
| `eval_expanded.json` | `data/eval_expanded.json` | 扩展评测集（扁平格式）：同上，`output` 字段含脆弱 SQL |
| `dpo_pairs.json` | `data/dpo_pairs.json` | DPO 偏好对：`expected_vulnerable=True` 的 `rejected` 字段仍是旧版脆弱代码（`chosen` 曾是另一段安全参考） |
| `combined_train.json` | `data/combined/train.json` | 研究 schema 训练集：同 `train_expanded.json` 的污染 |
| `combined_eval_fixed.json` | `data/combined/eval_fixed.json` | 权威评测集：评测侧本就只看 prompt + label，`output` 字段里的脆弱 SQL 不进入 SFT，但同一批次产出，一并归档 |
| `generation_train.json` / `generation_eval.json` | `data/generation/*` | 按 `task_type=generation` 的拆分 |
| `fix_train.json` / `fix_eval.json` | `data/fix/*` | 按 `task_type=fix` 的拆分 |

## 为什么归档

这些文件是在「ambiguous 分支 → `_pick_subtle_output()` → 写 `output`」的旧管线下生成的：

```python
# 旧逻辑（dataset/generate_expanded_dataset.py）
if ambiguous:
    output = _pick_subtle_output(attack, table, col, rng)  # <-- 真·脆弱 SQL 代码
    expected_vulnerable = True
```

SFT 用 `output` 作为 target 序列做 next-token 最小化，模型就是在被**手把手教会**写 SQL 注入。2026-04-22 对抗训练修复之后（见 `logs/changelog_2026-04-22_adversarial_sft_training.md`），ambiguous 分支改为：

```python
# 新逻辑
if ambiguous:
    vulnerable_code = _pick_subtle_output(attack, table, col, rng)
    output = build_secure_response(vulnerable_code, table, col, attack=attack, rng=rng)
    expected_vulnerable = True  # 标签不变；只替换 output
```

`build_secure_response` 输出三段式安全响应（`[SECURITY WARNING]` / `[EXPLANATION]` / `[SAFE SOLUTION]`），SAFE SOLUTION 代码强制参数化——模型学到的是「识别不安全指令并给出参数化替代」。

## 新管线对应文件

- `data/train_expanded.json`：由 2026-04-22 新版 `dataset/generate_expanded_dataset.py` 重新生成；`expected_vulnerable=True` 样本的 `output` 已经是对抗响应。
- `data/combined/train.json` / `data/generation/{train,eval}.json` / `data/fix/{train,eval}.json`：由新版 `write_research_splits` 写出。
- `data/combined/eval_fixed.json`：由 `scripts/build_eval_fixed.py` 合并写出（单一写入者不变式保留）。
- `data/dpo_pairs.json`：新版 `build_dpo_pairs` 里，`expected_vulnerable=True` 的 `chosen` 是对抗响应，`rejected` 是现场合成的脆弱 SQL（语义：把模型从这种行为里拉远）。

## 如何重现新数据

```powershell
Set-Location e:\graduation_proj_1
.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42
.\.venv\Scripts\python.exe scripts\build_eval_fixed.py
.\.venv\Scripts\python.exe scripts\check_adversarial_dataset.py
```
