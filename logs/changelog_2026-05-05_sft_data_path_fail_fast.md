# Changelog — 2026-05-05 SFT 数据路径静默回退移除（十九次加固）

> 移除 `train_lora_sft.py` 与 `train_qlora_sft.py` 中在扩展训练集缺失时
> 静默回退到仅有 ~100 条旧 schema 数据的逻辑。改为 FAIL FAST 并明确指引修复。

---

## 1. 背景与危害（Problem）

### Bug #11：SFT 训练数据路径解析为回退格式，不包含扩展数据集

**文件**：`training/train_lora_sft.py → main()` / `training/train_qlora_sft.py → main()`

**根因**：

旧版数据路径解析逻辑：
```python
train_json = files.get("train_sft_json")
if train_json and (ROOT / train_json).exists():
    data_path = ROOT / train_json
else:
    data_path = ROOT / files.get("sql_security_dataset", "dataset/sql_security_dataset.json")
```

| 条件                                  | 行为                                                      | 问题 |
| ------------------------------------- | --------------------------------------------------------- | ---- |
| `data/combined/train.json` 存在       | 正常加载扩展数据集（~2500 条）                            | ✓    |
| `data/combined/train.json` **不存在** | 静默回退到 `dataset/sql_security_dataset.json`（~100 条） | ✗    |

**回退数据的致命缺陷**：
1. **小样本量**：仅 ~100 条 vs 扩展集的 ~2500 条 → 训练严重欠拟合
2. **不同 schema**：使用 `category` 而非 `attack_type/difficulty/task_type/expected_vulnerable`
3. **Code-only 不兼容**：output 包含完整函数定义 + 教程文本，与新 code-only 契约不匹配
4. **完全静默**：无任何警告或错误——用户不知道训练在错误数据上进行

**后果**：
静默回退 → 微型数据集训练 → 严重欠拟合 → 评测指标差 → 用户可能归因于模型而非数据问题。

**严重性**：🟡 中（类别 A — 格式/解析）

---

## 2. 修复方案（Solution）

### 移除回退，改为 FAIL FAST

**`train_lora_sft.py`** 与 **`train_qlora_sft.py`** 两处同步修改：

```python
# 2026-05-05 修复（问题 #11）：移除静默回退
train_json = files.get("train_sft_json")
if not train_json:
    raise FileNotFoundError(
        "配置中缺少 files.train_sft_json；"
        "请检查 configs/default.yaml 中 files.train_sft_json 是否指向扩展训练集。"
    )
data_path = ROOT / train_json
if not data_path.exists():
    raise FileNotFoundError(
        f"训练数据 {data_path} 不存在。\n"
        "请运行: python dataset/generate_expanded_dataset.py --num_samples 2500\n"
        "然后将产出 data/combined/train.json 作为 files.train_sft_json 指向的目标。"
    )
```

### 两级检查

1. **配置缺失**：`train_sft_json` 键不在 `configs/default.yaml` 的 `files` 段 → `FileNotFoundError` + 配置指引
2. **文件不存在**：路径存在但 JSON 文件缺失 → `FileNotFoundError` + 数据生成命令

---

## 3. 影响范围（Impact）

| 文件                          | 变更                             | 影响                     |
| ----------------------------- | -------------------------------- | ------------------------ |
| `training/train_lora_sft.py`  | 移除回退逻辑，改为两级 FAIL FAST | 扩展训练集缺失时立即报错 |
| `training/train_qlora_sft.py` | 同上                             | 同上                     |

**不触碰**：
- `configs/default.yaml` — `files.train_sft_json` 配置不动
- `dataset/generate_expanded_dataset.py` 零改动
- `dataset/sql_security_dataset.json` 保留不动（供其他用途或降级调试）
- 评测管线零改动

---

## 4. 测试验证

全部 61 条既有测试通过，无回归：

```
============== 61 passed, 6 warnings, 4 subtests passed in 1.18s ==============
```

---

## 5. 相关文档

- `README.md` — 「关键修复（十九）」条目
- `training/train_lora_sft.py` — `main()` 数据加载段
- `training/train_qlora_sft.py` — `main()` 数据加载段
- `configs/default.yaml` — `files.train_sft_json`
