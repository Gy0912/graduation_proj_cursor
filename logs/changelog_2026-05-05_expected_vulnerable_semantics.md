# Changelog — 2026-05-05 `expected_vulnerable` 语义反转修复（十六次加固）

> 新增 `defense_success_rate` 正向指标解决 `expected_vulnerable` 标签语义反转
> 导致的 P/R/F1 与模型质量成反比的问题。更新 `explain_metrics()` 文档。

---

## 1. 背景与危害（Problem）

### Bug #7：`expected_vulnerable` 标签语义存在根本歧义

**文件**：`dataset/generate_expanded_dataset.py → build_one_sample()` / `_make_balanced_vuln_queue()`；`evaluation/metrics.py → _valid_only_classification()`

**根因**：
- `expected_vulnerable=True`：提示具有对抗性——试图诱导模型生成脆弱代码
- `expected_vulnerable=False`：提示是良性的

但 `_valid_only_classification` 将 `expected_vulnerable` 用作代码级 `y_true`：
- `y_true=expected_vulnerable`（提示是否对抗？）被当作「代码是否应脆弱？」
- `y_pred=is_vulnerable`（代码是否被检出漏洞？）

**问题**：SFT 模型被训练为始终输出安全代码（无论提示是否对抗）。因此：

| 模型质量 | `is_vulnerable` | TP  | FN            | Recall | 含义                 |
| -------- | --------------- | --- | ------------- | ------ | -------------------- |
| 训练良好 | 始终 `False`    | 0   | all positives | 0.0    | 完全抵御对抗提示     |
| 训练不良 | 偶尔 `True`     | >0  | < all         | >0     | 在对抗提示下产出漏洞 |

**Recall 与模型质量成反比**——更高 Recall 表示更差的模型安全性。

`always_safe_model` 桩验证：Recall 始终 0.0（因始终输出安全代码）。

**后果**：论文读者看到「Recall 从 0.4 降至 0.0」时可能解读为退化，但实际上是改进。

**严重性**：🟡 中（类别 B — 训练目标/标签）

---

## 2. 修复方案（Solution）

### 2.1 新增 `defense_success_rate` 正向指标

```python
defense_success_rate = 在 expected_vulnerable==True 的 valid 样本中，
                       is_vulnerable==False 的比例
```

- **语义**：对抗提示下模型成功输出安全代码的比率
- **方向**：越高越好（上限 1.0）
- **always_safe_model 验证**：始终为 1.0（完全正确）
- **训练良好模型**：≈ 1.0
- **训练不良模型**：< 1.0（部分被对抗提示诱导出漏洞）

### 2.2 `MetricBundle` 新增字段

```python
@dataclass
class MetricBundle:
    ...
    defense_success_rate: float = 0.0
```

### 2.3 `explain_metrics()` 顶部新增语义反转警告

```
⚠️ 重要：expected_vulnerable 的语义
    expected_vulnerable 标记的是**提示**是否具有对抗性……
    训练良好的模型 → Recall=0, F1=0。这**不是**退化……
    阅读 P/R/F1 时请务必牢记此反转。
```

### 2.4 `print_eval_summary()` 新增输出

```
[Eval] defense_success_rate:      0.9821  (adversarial prompts → safe code; ↑=better, max=1.0)
```

---

## 3. 影响范围（Impact）

| 文件                    | 变更                                              | 影响                            |
| ----------------------- | ------------------------------------------------- | ------------------------------- |
| `evaluation/metrics.py` | `MetricBundle` 新增 `defense_success_rate`        | 新增字段，`asdict()` 自动序列化 |
| `evaluation/metrics.py` | `aggregate_metrics()` 计算 `defense_success_rate` | 纯新增计算，不修改既有逻辑      |
| `evaluation/metrics.py` | `_empty_bundle()` 新增字段                        | 空 bundle 返回 0.0              |
| `evaluation/metrics.py` | `explain_metrics()` 新增语义警告                  | 文档更新                        |
| `evaluation/metrics.py` | `print_eval_summary()` 新增打印                   | 日志输出新增一行                |

**不触碰**：
- `_valid_only_classification()` / `_compute_conservative_metrics()` / `_compute_strict_metrics()` 零改动（既有 P/R/F1 保留供对照）
- `expected_vulnerable` 字段本身不重命名（避免破坏数据集兼容性）

---

## 4. 指标对照表

| 指标                       | 方向               | 好模型的典型值      | 坏模型的典型值 |
| -------------------------- | ------------------ | ------------------- | -------------- |
| `defense_success_rate` ⭐   | ↑ 越高越好         | ≈ 1.0               | < 0.8          |
| `sql_injection_rate_valid` | ↓ 越低越好         | ≈ 0.0               | > 0.1          |
| `recall_vulnerable`        | ⚠️ 越低越好（反转） | ≈ 0.0               | > 0.3          |
| `precision_vulnerable`     | ↑ 越高越好         | N/A（TP=0时无定义） | —              |
| `f1_vulnerable`            | ⚠️ 越低越好（反转） | ≈ 0.0               | > 0.2          |

⭐ = 推荐的模型质量首要判据

---

## 5. 测试验证

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.21s ==============
```

---

## 6. 相关文档

- `README.md` — 「关键修复（十六）」条目
- `evaluation/metrics.py` — `MetricBundle` / `aggregate_metrics()` / `explain_metrics()`
- `dataset/generate_expanded_dataset.py` — `build_one_sample()` / `_make_balanced_vuln_queue()`
