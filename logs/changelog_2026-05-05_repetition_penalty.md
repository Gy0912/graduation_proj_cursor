# Changelog — 2026-05-05 贪心解码重复惩罚（十二次加固）

> 为 `run_eval_on_prompts()` 的 `model.generate()` 调用添加 `repetition_penalty` 与
> `no_repeat_ngram_size` 参数，防止贪心解码（temperature=0）时的 token 退化循环。

---

## 1. 背景与危害（Problem）

### Bug #4：无重复惩罚的贪心解码

**文件/模块**：`evaluation/evaluator.py → run_eval_on_prompts()`（第 527-535 行）；`configs/default.yaml → generation` 段落

**根因**：
- `temperature=0` 意味着 `do_sample=False`（贪心解码），模型总是选取最高概率 token。
- `top_p` 在采样关闭时无效。
- 缺少 `repetition_penalty`：一旦模型开始重复，贪心解码会持续选取同一 token 直至 `max_new_tokens`。
- 缺少 `no_repeat_ngram_size`：无法在 n-gram 层面阻止循环。

**后果**：
贪心解码 + 无重复惩罚 → 模型在生成有效代码块后陷入 token 重复循环（例如无限重复
`return cur.fetchall()`），产生超过 256 token 的退化输出。这直接导致
`extracted_candidate=None`（退化文本无法通过 AST 解析），将有效代码生成变为抽取失败。

### 严重性：🟠 高（类别：D — 解码/推理）

---

## 2. 修复方案（Solution）

### 2.1 `evaluation/evaluator.py` — `run_eval_on_prompts()`

**签名变更**：在 `load_in_4bit` 之后新增两个可选参数，保持向后兼容：

```python
def run_eval_on_prompts(
    ...
    load_in_4bit: bool,
    repetition_penalty: float = 1.0,      # 新增
    no_repeat_ngram_size: int = 0,         # 新增
    ...
```

**`model.generate()` 调用**：透传参数，仅在非默认值时生效：

```python
out_ids = model.generate(
    ...
    repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
    no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size > 0 else None,
    ...
)
```

**新增日志**：评测启动时打印完整 generation 配置以便审计：

```
[eval] generation: max_new_tokens=256, temperature=0, top_p=0.9, repetition_penalty=1.05, no_repeat_ngram_size=0
```

### 2.2 `configs/default.yaml` — `generation` 段落

```yaml
generation:
  max_new_tokens: 256
  temperature: 0
  top_p: 0.9
  repetition_penalty: 1.05    # 温和惩罚，降低已出现 token 的概率
  no_repeat_ngram_size: 0     # 0=关闭；代码场景下 >0 过于激进
```

`repetition_penalty=1.05` 的设计考量：
- `1.0` = 无惩罚（等价于旧行为）
- `1.05` = 温和惩罚，降低已出现 token 的 logits，足以打破循环但不过度扭曲分布
- 代码中会自然重复 token（`def`, `return`, 变量名等），过高值（如 1.2）会影响代码质量
- `no_repeat_ngram_size=0`（关闭）：代码场景下禁止 n-gram 重复过于激进（合法的 `def foo():` / `return x` 等模式会被误杀）

### 2.3 `evaluation/evaluate.py` — `main()`

从配置读取并传参：

```python
repetition_penalty = float(gen.get("repetition_penalty", 1.0))
no_repeat_ngram_size = int(gen.get("no_repeat_ngram_size", 0))
```

### 2.4 为什么不用 `stopping_criteria`

`eos_token_id` 已正确设置。问题不在于缺少停止条件，而在于模型陷入循环后**不生成
EOS token**。`repetition_penalty` 从源头阻止循环发生，比事后停止更优。

---

## 3. 影响范围（Impact）

| 变更文件                  | 变更类型                          | 下游影响                               |
| ------------------------- | --------------------------------- | -------------------------------------- |
| `evaluation/evaluator.py` | `run_eval_on_prompts()` 签名+调用 | 新参数有默认值，现有调用者不受影响     |
| `configs/default.yaml`    | `generation` 段新增 2 个键        | 评测默认启用 `repetition_penalty=1.05` |
| `evaluation/evaluate.py`  | `main()` 读取+传参                | `.get()` 安全回退，旧配置兼容          |

**不触碰的部分**：
- 训练管线零改动（`training/` 下的 `model.generate()` 调用不涉及）
- 检测逻辑（Bandit / 规则 / 污点）零改动
- 指标聚合逻辑零改动
- `run_eval_always_safe()` 零改动

---

## 4. 参数调优指南

| 场景         | `repetition_penalty` | `no_repeat_ngram_size` |
| ------------ | -------------------- | ---------------------- |
| 默认（推荐） | 1.05                 | 0                      |
| 激进防重复   | 1.1                  | 3                      |
| 关闭（调试） | 1.0                  | 0                      |
| 自然语言生成 | 1.1-1.2              | 3-4                    |

---

## 5. 测试验证（Verification）

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.39s ==============
```

---

## 6. 相关文档

- `README.md` — 「关键修复（十二）」条目
- `configs/default.yaml` — `generation.repetition_penalty` / `generation.no_repeat_ngram_size`
- [HuggingFace GenerationConfig — repetition_penalty](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.repetition_penalty)
