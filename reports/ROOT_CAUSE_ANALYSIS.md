# LLM 代码安全对齐管线失败 —— 根因分析报告

> 分析日期：2026-05-10  
> 分析对象：SQL 注入检测 / 安全代码生成 LoRA-SFT-DPO 微调管线  
> 基座模型：StarCoder2-3B  
> 分析方法：代码审计 + 数据分布分析 + 评测指标逆向 + 训练动力学推理

---

# 1. Executive Summary

该管线存在 **三个相互独立但级联放大的根因**，按重要性排列：

| 优先级 | 根因                                                                                                                                                                                                                                          | 严重度 |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **P0** | **SFT 训练目标与评估指标的语义冲突**（code-only 训练使模型以纯 Python 代码为 target，但评估按 `expected_vulnerable` 标签期望模型「对安全 prompt 出安全代码、对恶意 prompt 出脆弱代码」——这是一个内在矛盾的目标函数）                          | 致命   |
| **P1** | **训练数据致命性低熵**（2200 条样本仅 25.7% 唯一 output，76% 使用 pymysql，top 2 模板占 13.8%。模型在 1 epoch 内即完全过拟合到固定模板集合，丧失泛化能力）                                                                                    | 致命   |
| **P2** | **DPO 偏好对构造的语义同构崩塌**（chosen/rejected 来自不同代码生成路径，使用不同的 import/driver/函数签名——DPO 学到的是「pymysql > SQLAlchemy」而不是「参数化 > 拼接」；仅 ~1100 个偏好对且 chosen/rejected 结构差异过大导致 log-ratio 爆炸） | 严重   |

**最核心的结论**：该项目在历史上经历了"先污染后消毒"的演化路径。最早的 SFT 数据直接包含脆弱 SQL 作为 training target（手把手教模型写注入）。此问题虽已被 28 轮 changelog 修复，但修复引入了一个更微妙的问题——**code-only 规范化使 SFT 训练目标坍缩为极低熵的模板集合，模型在 1 epoch 内过拟合到 ~565 个唯一模板，然后在评估时对所有 prompt（无论安全/恶意）以 ~80% 的概率从训练分布中采样——而这些采样的 token 恰好包含训练期间从 fix-task INPUT 中暴露的脆弱 token 模式。**

换句话说：**该项目意外训练了一个「脆弱代码模仿器」而非「安全代码生成器」。**

---

# 2. Failure Tree（完整因果链）

```
LEVEL 0: 上游设计错误
├── [E1] 最初设计将脆弱 SQL 写入 SFT output → model learns injection
├── [E2] expected_vulnerable 标签语义设计：标记 prompt 是否恶意
│         而非标记 output 是否应脆弱
└── [E3] fix-task INPUT 暴露完整脆弱代码给模型上下文窗口

LEVEL 1: 数据构造问题
├── [D1] 安全模板仅 ~565 个唯一形态（25.7% 唯一率）
├── [D2] 76% 输出使用 import pymysql → 单一 driver 过拟合
├── [D3] top-2 输出模板占 13.8%（158+145/2200）
├── [D4] fix-task INPUT 中 25% 含脆弱 SQL 模式
├── [D5] DPO 偏好对数量少（~1100 对），且 chosen/rejected 存在
│         import 级别的不对齐（跨 driver 对比）
└── [D6] 评测集 task 分布与训练集不匹配（eval 85% generation, 
          train 50% generation）

LEVEL 2: 训练动力学问题
├── [T1] SFT: 极低输出熵 → 1 epoch 内 loss 急速下降 → 完全过拟合
├── [T2] SFT: 过拟合模型的 token 概率分布高度集中，非训练模板 token 概率→0
├── [T3] DPO-from-SFT: chosen 和 rejected 均在过拟合 SFT 模型下
│         概率→0 → log-ratio 发散 → NaN grad → collapse
└── [T4] DPO-from-baseline: baseline 无安全/脆弱偏好区分能力
          → chosen≈rejected 概率 → log-ratio≈0 → loss 停滞 → 零学习

LEVEL 3: 评估问题
├── [V1] expected_vulnerable 语义反转：模型越安全 → recall 越低
├── [V2] code-only 训练 + 三段式 marker 评估 = 永久 mismatch
└── [V3] 评估 prompt 分布与训练 prompt 分布偏移（评测 prompt 中
          含训练未见过的高级对抗短语）
```

**级联放大链**：
```
E1+E2 → D1+D2+D3 → T1 → T2 → T3 → model collapse
                         → T4 → zero progress
E3 → D4 → T1 放大（模型同时记忆 input 中的脆弱词表）
```

---

# 3. Verified vs Unverified Hypotheses

对当前人类假设的逐条验证：

### H1: "SFT dataset flaw: fix-task INPUT contains full vulnerable code"

**→ VERIFIED but INCOMPLETE**

验证证据：
- `data/combined/train.json` 中 fix-task 的 `input_code` 字段包含完整脆弱 Python 代码（含 `cur.execute(q)` 等注入模式）
- 25.2%（277/1100）的 fix-task INPUT 触发 `contains_vulnerable_sql_pattern` 检测
- 评分：**部分正确**。INPUT 含脆弱代码是事实，但这本身不是问题——fix 任务的正确定义就是「看到脆弱代码→输出安全代码」。真正的问题是**模型没有学到 INPUT→OUTPUT 的映射，而是学到了 INPUT 中的 token 分布作为语言模型先验**。

### H2: "generation-task OUTPUT contains only ~19 safe templates"

**→ PARTIALLY VERIFIED（数量不完全准确，但方向正确）**

验证证据：
- 实际安全模板数量：565 个唯一 output（25.7% of 2200），而非 19 个
- 但 top-2 模板占 13.8%，top-10 占 ~26%
- 这意味着 2200 条样本中，约 572 条（26%）来自仅 10 个模板
- **核心问题不是模板总数太少，而是频率分布极度不均**——少数高频模板占据了不成比例的训练信号

### H3: "model learns vulnerable pattern memorization instead of security reasoning"

**→ VERIFIED**

验证证据：
- SFT 模型 sql_injection_rate_valid = **80.3%**（vs baseline 13.5%）
- SFT 模型 FPR = **81.3%**（安全 prompt 上也输出脆弱代码）
- 这意味着模型在所有 prompt 上以 ~80% 概率输出被检测器判定为「含 SQL 注入」的代码
- 这是典型的**分布坍缩 + 训练数据 token 分布偏差**

### H4: "DPO-from-SFT: chosen logprob < rejected logprob, log-ratio explodes, gradients NaN"

**→ VERIFIED（由 changelog #25-27 确认）**

验证证据：
- changelog #25: EOS 丢失 + completion boundary 错乱 → NaN loss
- changelog #26: DPO 数据语义损坏（变量引用 undefined）
- changelog #27: `# ref=` 噪声导致 logits 降至 -800 以下 → NaN → collapse
- 当前 DPO 评测结果显示模型已回到 baseline 水平（13.3% vs 13.5%），验证了"DPO 完全擦除 SFT 学习，且自身无增益"

### H5: "DPO-from-baseline: weak signal, loss stagnates"

**→ PARTIALLY VERIFIED**

验证证据：
- DPO 模型评测结果与 baseline **几乎完全相同**（sql_injection_rate: 13.3% vs 13.5%，混淆矩阵差异在 ±1 内）
- 这表明 DPO 或者完全未学到任何东西，或者学到的被后续 collapse 擦除
- **不完整之处**：当前 DPO 均从 SFT checkpoint 初始化，不存在真正的"DPO-from-baseline"。DPO 的零学习现象更可能是 SFT→DPO 过渡中的 collapse 而非 baseline 无偏好信号

---

# 4. Dataset Pathology Analysis

## 4.1 数据熵分析

```
指标                         | 值          | 健康基线   | 判定
Total samples                | 2,200       | ≥2000      | ✓
Unique outputs               | 565         | ≥1100      | ✗ 严重不足
Uniqueness ratio             | 25.7%       | ≥50%       | ✗ 致命低熵
Top-1 template frequency     | 7.2% (158)  | <2%        | ✗ 过度集中
Top-2 cumulative             | 13.8%       | <4%        | ✗
pymysql dominance            | 76.2%       | <40%       | ✗ 单一 driver
Task balance (gen:fix)       | 50:50       | 50:50      | ✓
ev balance (True:False)      | 50:50       | 50:50      | ✓
```

**解释**：25.7% 的唯一率意味着模型平均每个 output 模板被重复训练约 3.9 次。对于代码生成任务，这严重不足——模型学到的不是「如何生成安全代码」而是「记忆 565 个代码片段」。

## 4.2 脆弱 token 暴露分析

fix-task INPUT 包含的脆弱 SQL 模式（按频率排序）：

| 脆弱模式                          | fix-input 触发次数 | 风险 |
| --------------------------------- | ------------------ | ---- |
| `execute(... + ...)` (字符串拼接) | ~200               | 高   |
| `execute(f"...")` (f-string)      | ~50                | 高   |
| `.format()` SQL                   | ~50                | 高   |
| `text("..." + ...)` (ORM misuse)  | ~100               | 中   |
| `%s` 裸占位符误用                 | ~55                | 中   |

**关键洞察**：SFT 训练时，模型在 fix-task 的 prompt 上下文中反复看到诸如 `cur.execute(f"SELECT...")`、`"SELECT ... " + user_input` 等 token 序列。虽然 completion target 是安全代码，但 **prompt 中的 token 同样参与 self-attention 并更新模型参数**。在低熵输出分布下，模型学到的生成策略是：「从 prompt 中见过的 token 分布中采样」。由于 fix-task prompt 中 25% 包含脆弱 SQL token，模型将这些 token 内化为「合法输出词汇表」的一部分。

## 4.3 安全代码模板的低熵特征

安全模板的核心问题是**各模板之间的 token 重叠度极高**：

```
模板 A: import pymysql\n\n\ndef fetch_rows(conn, value):\n    sql = "SELECT * FROM X WHERE Y = %s"\n    cur.execute(sql, (value,))\n    return cur.fetchall()
模板 B: import sqlite3\n\n\ndef fetch_rows(conn, value):\n    sql = "SELECT * FROM X WHERE Y = ?"\n    cur.execute(sql, (value,))\n    return cur.fetchall()
```

两条模板共享约 85% 的 token。从模型视角看，学习目标几乎不可区分——唯一差异是 `pymysql` vs `sqlite3` 和 `%s` vs `?`。这导致：
1. 模型无法学到「何时用哪个 driver」
2. 模型把所有 driver 的 token 混入同一概率簇
3. `%s`（pymysql 参数化占位符）与 `%s`（出现在脆弱 `"%" + "%s" + "%"` 中的概率）在 token 空间中完全相同的，模型无法区分

## 4.4 训练/评估分布偏离

| 维度            | 训练                                   | 评估                    | 偏离度   |
| --------------- | -------------------------------------- | ----------------------- | -------- |
| generation 占比 | 50%                                    | 85%                     | **严重** |
| hard 占比       | 40%                                    | 45%                     | 轻微     |
| 对抗短语        | 10 种                                  | 12 种（含 2 种 unseen） | 中等     |
| prompt 格式     | `Instruction:\n...\n\nInput:\n...\n\n` | 同                      | ✓        |

评估集 85% 为 generation 任务，但训练集仅 50% generation。这意味着模型在评估时遇到的任务分布与训练时截然不同。对于已过拟合到训练分布的 SFT 模型，这会导致严重的性能退化。

## 4.5 Shortcut Learning 分析

**已确认的高风险快捷路径**：

1. **`import` 行作为任务判别器**：模型可能学会了「看到 `import pymysql` → 接 `def fetch_rows`」的简单模式匹配，而非理解安全语义
2. **函数名泄漏**：fix-task 的 output 保留了 INPUT 中的函数名（如 `def bad`、`def lookup`），模型可以简单地复制函数签名而非学习转换
3. **`# ref=` 注释**（已修复）：曾是模型记忆的「输出尾部标志」
4. **占位符 `%s` 的歧义**：同一个 token `%s` 在安全代码中是 pymysql 占位符，在脆弱代码中是 Python 字符串格式化操作符的组成部分——模型无区分能力

---

# 5. SFT Failure Analysis

## 5.1 为什么 SFT 模型输出脆弱代码？

**直接原因不是训练数据中的安全代码有问题，而是过拟合 + token 分布偏差的联合效应。**

机制分析：

**阶段 1 — 过拟合**：
- 2200 条样本仅 565 个唯一 output → 极低学习难度
- 默认 2 epoch → 模型在第一个 epoch 后半段 loss 已趋于 0
- 过拟合模型的 logit 分布极度尖锐：训练分布内 token 概率 ≈1.0，分布外 token 概率 ≈0

**阶段 2 — 评估时的分布外泛化**：
- 评估 prompt 使用了训练中未见的表名/列名组合 + unseen 对抗短语
- 过拟合模型面对分布外输入时，logit 分布坍缩为「最接近训练分布的 token」
- 由于 fix-task 训练将脆弱 SQL 的 token（`+`、`f"`、`"+"`、`'` 等）暴露在 prompt 中，这些 token 在模型的 residual stream 中已有非零表示

**阶段 3 — 脆弱 token 的激活优势**：
- 训练时，模型在 fix-task prompt 中反复看到脆弱模式（`"SELECT ... " + x`）
- 尽管 completion target 是安全的，但 attention 层已经将 prompt 中的脆弱 token 与 output 位置关联
- 评估时，模型生成的 token 序列天然偏向 prompt 中见过的模式 → 输出脆弱代码

**结论**：SFT 未能教会模型「安全」概念，而只教会模型「在特定 prompt 格式下输出特定的模板化代码」。当 prompt 偏离训练分布，模型回退到从 prompt 上下文（包含脆弱模式）中采样——输出脆弱代码。

## 5.2 是记忆、坍缩、快捷学习还是目标混淆？

**四者同时存在，但主因是目标混淆（Objective Confusion）**。

- **记忆**：部分正确——模型确实记忆了 565 个模板
- **坍缩**：部分正确——过拟合导致生成坍缩到少数模式
- **快捷学习**：正确——模型学会了 `import X → def Y → execute(Z, (W,))` 的表面模板
- **目标混淆**：**这是根因**。训练目标（最小化 next-token CE loss）与安全目标（不输出 SQL 注入）在 token 层面存在内在冲突——`%s` token 在安全代码中是奖赏信号（降低 loss），在脆弱代码中也可能是奖赏信号（如果 prompt 中有 `%s`）。模型无法在 token 预测层面区分「好的 %s」和「坏的 %s」。

## 5.3 指令格式是否导致问题？

**部分导致**。`Instruction:\n...\n\nInput:\n...\n\n` 格式本身没问题，但 fix-task 的 INPUT 中包含的 `Vulnerable code:\n```python\n...脆弱代码...\n```\n` 前缀向模型暴露了完整的脆弱代码。模型的学习信号是：

```
P(vulnerable_token | "Vulnerable code:\n```python\n...")  → 通过 attention 提升
P(safe_token | completion) → 通过 CE loss 降低
```

两者的梯度方向在 token 嵌入空间中存在竞争，而 CE loss 的梯度远强于 attention 中的关联学习 → 模型 'memorize' 了安全输出但 'internalize' 了脆弱词表。

---

# 6. DPO Failure Analysis

## 6.1 DPO-from-SFT 失败

**失败机制（逐步）**：

**Step 1 — SFT 遗产**：
SFT 模型过拟合到 565 个模板，logit 分布极度尖锐。对于 DPO 偏好对中的 chosen（安全代码）和 rejected（脆弱代码），SFT 模型的概率赋值如下：
- Chosen（安全代码）：若此安全模板在 SFT 训练中出现过（高频模板）→ 概率较高；若未出现过 → 概率极低（≈0）
- Rejected（脆弱代码）：在 SFT 训练中作为 fix-task INPUT 出现过 → 模型的 token 分布中有非零概率；且由于脆弱 token（`+`、`f"`) 在自然代码中更常见 → 概率可能反超 chosen

**Step 2 — Log-Ratio 爆炸**：
DPO loss：$\mathcal{L} = -\log\sigma\left(\beta \log\frac{\pi_\theta(c|x)}{\pi_{\text{ref}}(c|x)} - \beta \log\frac{\pi_\theta(r|x)}{\pi_{\text{ref}}(r|x)}\right)$

当 $\pi_{\text{ref}}(c|x) \to 0$（chosen 不在 SFT 分布中）时，$\log\frac{\pi_\theta(c|x)}{\pi_{\text{ref}}(c|x)}$ 发散。
当 $\pi_{\text{ref}}(r|x) > \pi_{\text{ref}}(c|x)$（rejected 概率更高）时，偏好方向反转。

**Step 3 — NaN 梯度**：
发散 log-ratio → sigmoid 输入为极大正/负值 → 梯度饱和或爆炸 → max_grad_norm 无法完全抑制 → NaN → 模型参数污染 → collapse

**此问题已被 changelog #25-28 部分修复**（降低 LR、增加 beta、修复 tokenization），但根本问题仍在——**只要 SFT 模型过拟合，DPO 就会在低概率 chosen 上崩溃**。

## 6.2 DPO-from-Baseline 停滞

**失败机制**：

Baseline 模型（StarCoder2-3B）未经过任何安全微调。其对安全代码和脆弱代码的概率几乎相等——两者都是「合法 Python 代码」。DPO loss 的信号强度取决于：

$$\Delta = \beta\left(\log\frac{\pi_\theta(c|x)}{\pi_{\text{ref}}(c|x)} - \log\frac{\pi_\theta(r|x)}{\pi_{\text{ref}}(r|x)}\right)$$

当 $\pi_{\text{ref}}(c|x) \approx \pi_{\text{ref}}(r|x)$（baseline 无偏好），初始 $\Delta \approx 0$。学习信号仅来自 $\pi_\theta$ 的微小随机波动。在 batch=1、grad_accum=8、LR=2e-7 的设置下，梯度噪声完全淹没信号 → 零学习。

**实证证据**：DPO 评测结果与 baseline 几乎完全一致（sql_injection_rate: 13.3% vs 13.5%），表明 DPO 未产生任何可测量的学习效果。

## 6.3 偏好对可分性分析

DPO 偏好对的核心问题是 **chosen 和 rejected 的代码结构差异过大**：

```
Chosen (safe):     import sqlite3  →  cur.execute(sql, (value,))  →  参数化
Rejected (vuln):   import pymysql  →  cur.execute(f"SELECT...")   →  f-string 注入
```

模型学到的信号：**sqlite3 > pymysql**（工具选择），而非**参数化 > 拼接**（安全选择）。这是 P0-2 修复（changelog #22）试图解决的问题，但即使在 AST 同构变换后，token 级别的差异仍然巨大。

---

# 7. Most Likely Primary Root Cause

## 7.1 SINGLE Highest-Impact Issue

**训练数据的致命低熵**（25.7% 唯一率，76% pymysql 集中度）。

这是**所有下游失败的共同上游**：

```
低熵数据 → SFT 过拟合（1 epoch 完成）→ token 概率分布坍缩
    ↓
    ├→ DPO chosen 概率→0 → log-ratio 爆炸 → NaN collapse
    ├→ 评估时分布外泛化失败 → 输出从 prompt 中见过的脆弱 token
    └→ 模型未学到「安全」概念 → 仅学到模板复制
```

**为什么其他候选不是 primary root cause**：

- **原始数据污染**：已被修复（output 已全部净化），且即使 output 干净，低熵问题仍存在
- **fix-task INPUT 含脆弱代码**：这是正确设计（fix 任务需要看到脆弱代码），问题在于低熵使模型无法区分 prompt 和 completion 的 token 分布
- **DPO tokenization 错误**：已被修复，修复后 DPO 仍无学习效果，说明问题更深层

## 7.2 Secondary Issues

| 优先级 | 问题                                 | 影响                           |
| ------ | ------------------------------------ | ------------------------------ |
| S1     | fix-task 脆弱 token 暴露在 prompt 中 | 放大低熵过拟合的后果           |
| S2     | `expected_vulnerable` 语义设计错误   | 使评估指标无法正确反映安全性能 |
| S3     | DPO 偏好对结构不对齐                 | 降低 DPO 信号质量              |
| S4     | 训练/评估分布偏移                    | 使过拟合后果在评估中被放大     |
| S5     | code-only 训练 + 三段式评估 mismatch | 评估契约内部不一致             |

## 7.3 Amplifying Issues

| 放大器                    | 机制                                           |
| ------------------------- | ---------------------------------------------- |
| 2-epoch SFT 训练          | 在低熵数据上将过拟合从「较多」放大到「完全」   |
| 高 learning_rate (2e-4)   | 加速过拟合收敛                                 |
| DPO beta=0.1（旧值）      | KL 约束太弱，无法阻止过拟合 SFT → DPO 分布崩溃 |
| max_grad_norm=1.0（旧值） | 无法在 NaN 前截断梯度                          |

## 7.4 Misleading Symptoms

| 症状                    | 表面解释                | 实际机制                                                                                                  |
| ----------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------- |
| SFT 输出 80% 脆弱代码   | "模型学会了 SQL 注入"   | 模型过拟合到低熵模板集，评估时分布外泛化失败，回退到 prompt 中见过的脆弱 token 分布                       |
| DPO 与 baseline 相同    | "DPO 无效果"            | DPO 过程中梯度 NaN → collapse → 参数被破坏 → 模型回退到 baseline 水平（或更差）                           |
| baseline 低脆弱率 (13%) | "baseline 已经比较安全" | baseline 只是生成无意义的通用代码，碰巧不含 SQL 注入语法；不是「学会了安全」，而是「没学会生成 SQL 代码」 |

---

# 8. Recommended Validation Experiments

### Experiment 1: 验证「低熵是主因」
**修改**：将安全输出模板从 565 个扩展到 ≥2000 个唯一形态（引入更多 driver、更多函数命名、更多代码结构变体），保持其他条件不变
**预期结果**：SFT sql_injection_rate 显著降低；DPO 开始产生正向学习信号
**解释**：若改善 → 确认低熵是主因；若无改善 → 需要检查其他因素

### Experiment 2: 验证「fix-task INPUT 污染」
**修改**：将 fix-task INPUT 中的脆弱代码替换为抽象描述（如 "The code below uses string concatenation to build SQL queries"），不展示实际脆弱 token
**预期结果**：SFT 模型脆弱输出率降低
**解释**：若改善 → 确认 prompt 中脆弱 token 暴露是关键通路

### Experiment 3: 验证「评估语义反转」
**修改**：将评估改为直接测量 `defense_success_rate`（对抗 prompt 上输出安全代码的比例），而非基于 `expected_vulnerable==is_vulnerable` 的 confusion matrix
**预期结果**：一个始终输出安全代码的完美模型获得 defense_success_rate=1.0
**解释**：确认当前评估框架的语义问题

### Experiment 4: 验证「DPO 从 healthy SFT 起始」
**修改**：从训练 0.5 epoch（而非 1-2 epoch，此时尚未完全过拟合）的 SFT checkpoint 启动 DPO
**预期结果**：DPO 产生可测量的学习进步
**解释**：若改善 → 确认 DPO 失败是因为 SFT 过拟合导致；若无改善 → DPO 有独立于 SFT 的问题

### Experiment 5: 验证「模板频率偏差」
**修改**：对训练集做 importance sampling 或 temperature sampling，使每个唯一模板的出现频率均匀化
**预期结果**：SFT 不再对高频模板过拟合
**解释**：确认频率不均的影响

---

# 9. Recovery Strategy

## 9.1 Immediate Fixes（立即可做）

| #   | 操作                                                     | 优先级 |
| --- | -------------------------------------------------------- | ------ |
| F1  | SFT epoch 保持在 1（完成）                               | P0 ✓   |
| F2  | SFT LR 保持在 1e-4（完成）                               | P0 ✓   |
| F3  | DPO beta 保持在 0.5（完成）                              | P0 ✓   |
| F4  | DPO max_grad_norm 保持在 0.5（完成）                     | P0 ✓   |
| F5  | 添加早停（early stopping）：val loss 连续 N 步不降即停止 | P0     |
| F6  | 添加梯度/激活 NaN 检测的更多守卫点                       | P1     |

## 9.2 Medium-Term Redesign（1-2 周）

| #   | 操作                                                                                                                                                              | 优先级 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| R1  | **扩展安全模板库**：从当前的 ~4 种核心安全模板（pymysql/sqlalchemy/sqlite/indirect_chain）扩展到 ≥50 种（增加不同函数名、类封装、上下文管理器、装饰器、异步模式） | P0     |
| R2  | **增加 driver 多样性**：引入 psycopg2、mysql-connector、aiomysql、asyncpg 等更多 driver                                                                           | P0     |
| R3  | **统一 DPO 偏好对的 driver**：确保 chosen 和 rejected 使用完全相同的 import 和 driver，仅 SQL 构造方式不同                                                        | P1     |
| R4  | **重构 fix-task INPUT**：将脆弱代码从纯文本暴露改为 token-masked 版本（用占位符替换最脆弱的 token）                                                               | P1     |

## 9.3 Dataset Reconstruction Strategy

核心原则：**高熵、多 driver、同构偏好对**。

```
目标数据分布：
- 唯一 output 模板：≥2000（唯一率 ≥90%）
- Driver 分布：pymysql 25% / sqlite3 20% / sqlalchemy 20% / psycopg2 15% / 其他 20%
- 代码结构多样性：函数/类/上下文管理器/装饰器/异步 各 ≥15%
- 变量名/函数名多样性：从固定名称池中随机采样 ≥100 个不同名称
- DPO 偏好对：chosen 和 rejected 必须共享 import + 函数签名 + driver
```

## 9.4 DPO Redesign Strategy

1. **从 early-stopping checkpoint（而非最终过拟合 checkpoint）启动 DPO**
2. **确保每个偏好对中的 chosen 和 rejected 是 AST 级别的同构变体**（仅 SQL 构造不同）
3. **偏好对 hardness 分层**：easy（明显错误 vs 明显正确）、medium（微妙错误 vs 正确）、hard（几乎正确 vs 完全正确）
4. **监控 log-ratio 分布**：若 log-ratio 超过 ±10 即暂停训练
5. **考虑替代损失函数**：若 DPO 持续不稳定，考虑 KTO 或 SLiC 作为备选

## 9.5 Evaluation Redesign Strategy

1. **将 `defense_success_rate` 作为主要正向指标**（越高越好）
2. **将 `sql_injection_rate_valid` 作为辅助安全指标**（越低越好）
3. **将 `full_compliance_rate`（三段式 marker 命中率）与训练契约对齐**：code-only 训练 → 预期为 0；对抗训练 → 预期接近 1
4. **添加 per-attack-type 的 breakdown**（已完成）
5. **添加 generation-only 和 fix-only 的子集指标**以隔离任务类型影响

---

# 10. Critical Design Principles Violated

| 原则                       | 违反描述                                                                                                        | 后果                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **训练-评估对齐**          | SFT 以纯 Python 代码为 target，评估期望三段式结构化输出 + 安全行为。训练目标和评估目标在多个维度上不一致        | 评估结果无法反映训练质量       |
| **数据多样性最低阈值**     | 安全代码生成需要模型理解「安全」的抽象概念，但训练数据仅 25.7% 唯一率，提供了 565 个具体实例而非 565 个概念示例 | 模型记忆实例而非学习概念       |
| **偏好对结构同构性**       | DPO 的 chosen 和 rejected 使用了不同的 import、driver 和函数签名。偏好信号被工具选择噪声淹没                    | DPO 学到错误的维度             |
| **标签语义一致性**         | `expected_vulnerable` 在训练中标记 prompt 是否对抗、在评估中被用来与 output 检测结果比对——两个语义不同          | 混淆矩阵数学上有效但语义上无效 |
| **避免 Shortcut Features** | fix-task INPUT 包含脆弱代码的完整 token 序列，为模型提供了复制 prompt 模式的快捷路径                            | 模型学会模式复制而非安全推理   |
| **KL 约束适当性**          | DPO beta=0.1 对过拟合 SFT 模型太弱，无法约束分布偏移                                                            | 分布崩溃                       |
| **梯度稳定性**             | 在已知 log-ratio 可能发散的情况下，缺乏 logit clamp 和 grad norm 监控（已在后期 fix 中添加）                    | NaN 污染模型参数               |
| **评测单一写入者**         | 评估集曾有两个来源，导致标签不一致（已在 fix #3 中修复）                                                        | 历史评估结果不可比             |
| **静默回退禁止**           | SFT 训练入口在数据缺失时曾静默回退到不同 schema 的小数据集（已在 fix #19 中修复）                               | 训练在错误数据上静默完成       |

---

# 附录 A: 关键数据摘要

| 指标                      | Baseline | LoRA-SFT  | LoRA-DPO |
| ------------------------- | -------- | --------- | -------- |
| sql_injection_rate_valid  | 13.5%    | **80.3%** | 13.3%    |
| safe_rate_valid           | 86.5%    | 19.7%     | 86.7%    |
| precision_vulnerable      | 53.8%    | 48.7%     | 55.3%    |
| recall_vulnerable         | 14.5%    | **79.3%** | 14.7%    |
| FPR (false positive rate) | 12.5%    | **81.3%** | 12.0%    |
| extraction_failure_rate   | 3.7%     | 5.3%      | 5.0%     |
| n_valid / n_total         | 289/300  | 284/300   | 285/300  |

**关键洞察**：
- SFT 使 FPR 从 12.5% → 81.3%（安全 prompt 上输出脆弱代码的概率增加 6.5 倍）
- DPO 使所有指标回到 baseline 水平——完全擦除 SFT 效果，且自身无增益
- 三个模型的 extraction_failure_rate 均在 3-5%，说明代码抽取本身不是主要问题

# 附录 B: 训练数据分布

```
Total:        2,200 samples
Generation:   1,100 (50%)    Fix: 1,100 (50%)
ev=True:      1,100 (50%)    ev=False: 1,100 (50%)

vulnerability_type:
  indirect_injection:    726 (33.0%)
  orm_misuse:            396 (18.0%)
  fake_sanitization:     396 (18.0%)
  parameterized_query:   220 (10.0%)
  format_string:         176 ( 8.0%)
  fstring:               176 ( 8.0%)
  string_concat:         110 ( 5.0%)

Driver usage in outputs:
  pymysql:    1,677 (76.2%)
  sqlalchemy:   326 (14.8%)
  sqlite3:      197 ( 9.0%)

Output uniqueness: 565/2200 (25.7%)
Top template freq: 158/2200 (7.2%)
```

---

*分析结束。建议优先执行 §8 中的 Experiment 1 和 Experiment 4 以验证根因假设，然后按 §9 中的优先级顺序逐步实施恢复策略。*
