# Changelog — 2026-05-05 训练-评测响应格式对齐（十三次加固）

> 修复 SFT code-only 训练目标与评测三段式响应质量 marker 之间的内部不一致。
> 新增 `code_only_training` 参数使评测指标正确反映训练契约。

---

## 1. 背景与危害（Problem）

### Bug #3：训练目标（纯代码）与评测的响应质量 Marker 不匹配

**文件/模块**：
- 训练数据：`dataset/generate_expanded_dataset.py → _make_safe_sft_output()` — 始终产出安全 Python 代码
- SFT 预处理：`training/sft_preprocess.py → normalize_sft_records_for_training()` — 抽取 code-only，丢弃三段式文本
- 评测指标：`evaluation/metrics.py → _compute_response_quality_metrics()` — 期望 `[SECURITY WARNING]`/`[EXPLANATION]`/`[SAFE SOLUTION]`

**根因**：

| 组件                                   | 行为                                                                     | 对 marker 的态度 |
| -------------------------------------- | ------------------------------------------------------------------------ | ---------------- |
| `normalize_sft_records_for_training()` | 从 output 中仅抽取 `[SAFE SOLUTION]` 内 Python，丢弃 warning/explanation | 主动剥离         |
| `FORBIDDEN_TRAINING_TOKENS`            | 训练 completion 中禁止出现三段式 marker 字面量                           | 主动禁止         |
| `_compute_response_quality_metrics()`  | 统计 `"[SECURITY WARNING]" in raw_output` 等命中率                       | 期望命中         |

**这三者构成逻辑矛盾**：
- 训练端：模型被明确教为**不**输出三段式 marker
- 评测端：指标衡量模型**是否**输出了三段式 marker
- 结果：`full_compliance_rate_on_positives` 在训练良好的 SFT 模型上**始终为 0.0**

**严重性**：🔴 严重（类别 B — 训练目标/标签）

**后果**：评测 JSON 中的 `response_quality_metrics.full_compliance_rate_on_positives=0.0`
被解读为"模型未学会三段式输出"，但模型明确被训练为不输出三段式。这是一项彻底的内部不一致。

---

## 2. 修复方案（Solution）

**设计原则**：不改变训练端（训练目标是 code-only，这是正确的设计选择），而是让评测端
**意识**到当前评测的是 code-only 模型，从而正确解读响应质量指标。

### 2.1 新增 `code_only_training` 参数

该布尔参数流经以下路径：

```
configs/default.yaml (eval.code_only_training)
  → evaluate.py::main() 读取
    → run_eval_on_prompts(code_only_training=...)
      → aggregate_metrics(code_only_training=...)
        → _compute_response_quality_metrics(code_only_training=...)
```

### 2.2 `evaluation/metrics.py`

**`_compute_response_quality_metrics()`** 新增 `code_only_training: bool = False` 参数：

- 返回 dict 新增 `"training_mode"` 字段（`"code_only"` 或 `"adversarial"`）
- `code_only_training=True` 时 note 标注：
  > Code-only training mode: 模型被训练为输出纯 Python 代码……所有 marker 命中率**预期为 0.0**——这不是质量缺陷，而是训练契约的正确结果。
- `code_only_training=False` 时 note 保持原有对抗训练契约文案

**`aggregate_metrics()`** 新增同名 keyword-only 参数并透传。

**`print_eval_summary()`** 根据 `training_mode` 字段切换打印文案。

### 2.3 `evaluation/evaluator.py`

- `run_eval_on_prompts()` 签名新增 `code_only_training: bool = True` → 透传 `aggregate_metrics()`
- `run_eval_always_safe()` 签名新增 `code_only_training: bool = True` → 透传

默认 `True` 是因为：默认配置为 SFT（code-only），符合大多数使用场景。

### 2.4 `evaluation/evaluate.py`

```python
code_only_training = bool(ev_cfg.get("code_only_training", True))
```

安全回退：旧配置文件无此键时默认 `True`（向后兼容）。

### 2.5 `configs/default.yaml` / `configs/default_run.yaml`

```yaml
eval:
  code_only_training: true  # SFT=code-only，marker 命中率预期全 0.0
```

---

## 3. 影响范围（Impact）

| 变更文件                   | 变更类型                                                    | 下游影响                                      |
| -------------------------- | ----------------------------------------------------------- | --------------------------------------------- |
| `evaluation/metrics.py`    | `_compute_response_quality_metrics()` 新增参数              | 返回 dict 新增 `training_mode` 字段，既有无损 |
| `evaluation/metrics.py`    | `aggregate_metrics()` 新增 keyword-only 参数                | 默认 `False`，直接调用者不受影响              |
| `evaluation/evaluator.py`  | `run_eval_on_prompts()` / `run_eval_always_safe()` 新增参数 | 默认 `True`，通过 `evaluate.py` 可控          |
| `evaluation/evaluate.py`   | 读取 `code_only_training` 并传参                            | `.get()` 安全回退，旧配置兼容                 |
| `configs/default.yaml`     | `eval` 段新增 `code_only_training`                          | 新增键，不影响既有键读取                      |
| `configs/default_run.yaml` | 同上                                                        | 同上                                          |

**不触碰的部分**：
- 训练管线零改动（`training/sft_preprocess.py` / `dataset/` 无变更）
- 检测逻辑零改动
- `_response_structure_flags()` 零改动（marker 检测逻辑不变）
- per-sample 字段 `has_warning` / `has_explanation` / `has_safe_solution` 照常写入

---

## 4. 使用指南

| 场景                       | `code_only_training` | 含义                                                |
| -------------------------- | -------------------- | --------------------------------------------------- |
| SFT / LoRA-SFT / QLoRA-SFT | `true`（默认）       | 模型输出纯代码，响应质量指标预期全 0.0              |
| DPO（对抗训练）            | `false`              | 模型输出三段式对抗响应，正样本高合规 / 负样本低合规 |
| Baseline（未训练基座）     | `true`               | 基座模型不会被训练产出三段式                        |
| always_safe_model          | `true`               | 恒安全桩不产出三段式                                |

在 `configs/default.yaml` 中按需设置：

```yaml
eval:
  code_only_training: false  # 对抗训练 / DPO 场景
```

---

## 5. 测试验证（Verification）

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 2.51s ==============
```

`aggregate_metrics()` 的 `code_only_training` 默认值为 `False`，因此直接调用
`aggregate_metrics(samples)` 的测试保持原有对抗训练语义，不受影响。

---

## 6. 相关文档

- `README.md` — 「关键修复（十三）」条目
- `training/sft_preprocess.py` — `normalize_sft_records_for_training()` / `FORBIDDEN_TRAINING_TOKENS`
- `dataset/adversarial.py` — `extract_code_only_completion()` / `_collapse_identical_halves()`
- `evaluation/metrics.py` — `_compute_response_quality_metrics()` / `aggregate_metrics()`
- `configs/default.yaml` — `eval.code_only_training`
