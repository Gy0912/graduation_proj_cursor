# Changelog — 2026-05-05 DPO 仅对抗提示生成（十七次加固）

> `build_dpo_pairs()` 跳过良性提示（`expected_vulnerable=False`），仅对
> 对抗提示（`expected_vulnerable=True`）生成 DPO 偏好对。聚焦偏好优化于
> 真正需要安全强化的场景。

---

## 1. 背景与危害（Problem）

### Bug #8：DPO Prompt 缺少对抗上下文——所有对被统一对待

**文件**：`dataset/generate_expanded_dataset.py → build_dpo_pairs()`

**根因**：每个训练行生成一个 DPO 对，无论 `expected_vulnerable` 值为何。
Prompt 为 `training_prompt(instruction, input)`——与 SFT 格式相同。

| 提示类型 | `expected_vulnerable` | 当前 DPO 行为       | 问题                             |
| -------- | --------------------- | ------------------- | -------------------------------- |
| 对抗性   | `True`                | 安全代码 > 脆弱变体 | 梯度有用，但未强化对抗识别       |
| 良性     | `False`               | 安全代码 > 脆弱变体 | SFT 已教会安全输出，DPO 信号冗余 |

**后果**：
- 良性提示上的 DPO 对为**冗余信号**：SFT 已在这些提示上教会模型输出安全代码，
  偏好优化不产生额外信息增益。
- DPO 在所有样本上施加相同的「更安全 > 更不安全」梯度，浪费计算资源于
  模型已掌握的场景，而非集中于模型最需要强化的对抗场景。

**严重性**：🟡 中（类别 B — 训练目标/标签）

---

## 2. 修复方案（Solution）

### 核心变更：仅对抗提示生成 DPO 对

在 `build_dpo_pairs()` 循环中，通过验证后立即跳过良性行：

```python
if not r["expected_vulnerable"]:
    benign_skipped += 1
    continue
```

### 日志输出

```
[DPO] pairs generated: 412 (adversarial only); benign skipped: 388/800 (48.50%)
```

### 设计理由

1. **良性提示**：SFT 已教会模型输出安全代码。在这些提示上 DPO 偏好的边际收益
   接近于零——模型几乎不会在这些提示上产出脆弱代码。

2. **对抗提示**：模型在这些提示上最容易犯错（被诱导生成脆弱代码）。DPO 的
   「安全 > 脆弱」信号在此处价值最高——直接强化模型在压力下的安全行为。

3. **计算效率**：减少 DPO 对数量 → 加快 DPO 训练速度，同时保持信号质量。

---

## 3. 影响范围（Impact）

| 文件                                   | 变更                                                  | 影响                                   |
| -------------------------------------- | ----------------------------------------------------- | -------------------------------------- |
| `dataset/generate_expanded_dataset.py` | `build_dpo_pairs()` 跳过 `expected_vulnerable==False` | DPO 对数量减少约 50%（取决于数据分布） |
| `dataset/generate_expanded_dataset.py` | 新增 `benign_skipped` 计数器 + 日志                   | 仅日志输出                             |

**不触碰**：
- DPO 训练管线零改动（`training/` 下的 DPO 训练代码不变）
- DPO 对结构零改动（chosen/rejected 字段语义不变）
- SFT 训练零改动

**DPO 训练注意事项**：DPO 对数量减少后，可能需要相应调整
`num_train_epochs_dpo` 或 `learning_rate_dpo` 以补偿减少的数据量。

---

## 4. 测试验证（Verification）

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.21s ==============
```

---

## 5. 相关文档

- `README.md` — 「关键修复（十七）」条目
- `dataset/generate_expanded_dataset.py` — `build_dpo_pairs()`
- `configs/default.yaml` — `training.num_train_epochs_dpo`
