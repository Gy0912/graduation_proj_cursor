# DPO collapse 根因分析与修复（2026-05-09）

## 问题表现

- DPO 训练在 step~60 崩溃
- gradient 反常升高并波动
- logits 降至 -800 以下
- 评估输出 `,%s, %s, %s, ...` 重复 token
- 所有评估样本 fail（`extract_python_code:invalid`）
- 模型输出包含 `# ref=602718913`——训练数据泄露

## 根因分析

### P0-A：`# ref=` 噪声 token 导致训练数据泄露

每个 vulnerable 代码模板末尾都有 `# ref={随机数}`：
```python
cur.execute(q)
return cur.fetchall()
# ref=602718913
```

这些 `# ref=` 注释在 **fix 任务的 INPUT** 中出现——模型在 SFT 训练中反复看到它们，
将其作为「输出代码的尾部模式」学习。模型 collapse 时优先输出最近记忆的训练数据
模式——即 `# ref=...` + 随后的 `%s` 重复序列。

**修复**：删除全部 7 个 `_vuln_*` 函数中的 `# ref=` 注释，并将 `# ref=` 加入
`FORBIDDEN_TRAINING_TOKENS`。

### P0-B：DPO max_grad_norm 过高

`max_grad_norm=1.0` 在 step 60 附近无法抑制梯度爆炸。DPO loss 的 log-ratio
在 chosen/rejected 差异较大时会产生极值梯度。

**修复**：`max_grad_norm` 降至 `0.5`。

### P0-C：SFT 训练数据含破坏 SQL 语义的 fix

`_safe_fix_from_vulnerable` 将 `WHERE col=%s AND status=%s` 硬编码为
`WHERE col=%s`，丢弃 AND 子句。模型学到「修复=破坏」。

**修复**（已在上一轮完成）：新增 `_extract_sql_template_from_vuln` 保留完整 SQL 结构。

### 修复链

```
# ref= 泄露 → SFT 模型记忆「代码末尾 = # ref= + %s 重复」
    ↓
SFT checkpoint 已污染
    ↓
DPO 训练在此 checkpoint 上
    ↓
grad_norm=1.0 → step 60 梯度突破 → logits NaN
    ↓
模型 collapse → 输出记忆训练数据片段
    ↓
评估端 extract_python_code 无法解析 → 100% invalid
```

## 为什么修复后不会再崩

| 修复                    | 机制                                                     |
| ----------------------- | -------------------------------------------------------- |
| 删除 `# ref=`           | 消除训练数据特有的尾部模式——模型不会记忆「尾接 %s 重复」 |
| `# ref=` 加入 forbidden | SFT 训练 pre-flight 检查拦截任何含 ref 的 output         |
| `max_grad_norm=0.5`     | 更严格梯度裁剪，step 60 处梯度被限制在安全范围           |
| SQL 模板保存（上轮）    | fix task 输出语义正确，不引入错误学习信号                |
| BOS token（上轮）       | SFT 训练-推理分布一致                                    |

## 改为的文件

- `dataset/generate_expanded_dataset.py`：删除全部 `# ref={salt}` 行
- `training/sft_preprocess.py`：`# ref=` 加入 `FORBIDDEN_TRAINING_TOKENS`
- `training/dpo_train.py`：`max_grad_norm` 0.5

## 兼容性

- 旧 checkpoint：**需重新生成数据 + SFT + DPO**
- Eval pipeline：不受影响
- 其他模型：兼容
