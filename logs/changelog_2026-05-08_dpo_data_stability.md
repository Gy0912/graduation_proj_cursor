# DPO 数据生成与训练稳定性全面修复（2026-05-08）

**严重级别**：P0（DPO 对语义损坏 → 训练 NaN/梯度爆炸/collapse）  
**影响范围**：`dataset/generate_expanded_dataset.py`

## 修复清单

### Task 1 (P0)：`_extract_likely_param()` 重写为 whitelist-first + AST 数据流追踪

**根因**：旧版使用黑名单排除法识别 SQL 参数名。SQL 中间变量（`frag`/`part`/`clause`/
`base`/`prefix`/`suffix`/`w`/`u`/`fmt`/`mid`/`q`）漏过黑名单 → 被误识别为用户输入 →
`safe fix` 生成引用未定义变量的 chosen → DPO 对语义损坏 → 训练在 early steps
梯度爆炸 → NaN logits。

**修复**：whitelist-first 策略
1. 从函数签名提取参数列表，最后一个参数是用户输入候选
2. 在 `execute()` 中直接匹配 → 命中则返回
3. 若不命中：追踪 `execute()` 第一个参数（SQL 变量）到其赋值语句，
   检查 RHS 是否包含候选参数 → 间接匹配
4. 回退：返回最后一个参数

**验证**：20/20 攻击×难度组合，全部正确提取用户输入参数名。

### Task 2 (P0)：新增 `_validate_dpo_pair_structure()` 校验

**位置**：`build_dpo_pairs()` 写入 `dpo.append` 前

**校验**：
- AST 可解析（chosen + rejected）
- `execute()` 参数引用的变量均在函数签名中定义
- 不允许明显未定义变量

**行为**：`skip + 记录统计 + 输出原因`（非 raise 崩溃）

**验证**：12/12 有效对通过，手工构造的损坏对正确拦截。

### Task 3 (P1)：增强 DPO 数据稳定性（防御性守卫）

以下守卫已在之前的修复中就位：
- `chosen == rejected` → skip（P2-10）
- `chosen/rejected` 为空 → skip + log（P1-6）
- `chosen` 脆弱模式命中 → raise（SFT 输出必须安全）
- `rejected` 未命中脆弱模式 → raise（DPO 负例必须可检出）
- `stable_dpo_trainer._prepare_dataset()` 重写（2026-05-08）：EOS 追加、合并 tokenize、拆分

新增：
- `_validate_dpo_pair_structure` → skip + log（Task 2）
- 统计输出：structural skip count + reasons

### Task 4 (P1)：SFT/DPO 数据一致性

**分析**：`generate_expanded_dataset.py` 使用单一 seed 一次生成全部数据
（train/eval/dpo），内部一致。不存代码级不一致。

**工作流风险**：若用户先训练 SFT 后重新运行 Step 4，regenerate 的 DPO pairs
来自新数据。但这属于工作流纪律问题（应先运行 Step 4 再 Step 9-10），非代码 bug。

### Task 5 (P1)：训练稳定性

已在之前修复中确认：
- `beta`：全部配置从 0.01 → 0.1
- `max_grad_norm`：1.0
- EOS handling：`stable_dpo_trainer._prepare_dataset` 重写后正确追加 EOS
- empty completion：守卫丢弃 + warning
- tokenizer special token：prompt `add_special_tokens=True`，合并 tokenize 保证边界一致

### Task 6：跨模型兼容性

所有修复基于标准 AST 分析/NLP tokenizer API，无硬编码模型名或 tokenizer 特定
hack。`_detect_driver_from_code` 使用通用词法检测（`sqlalchemy`/`sqlite3`/
`pymysql` 关键字）。StarCoder2 / DeepSeek-Coder / Qwen-Coder 均可运行。

## 兼容性矩阵

| 维度           | 影响                                             |
| -------------- | ------------------------------------------------ |
| Dataset schema | 无变更（`chosen_framework` 字段已在 P0-4 新增）  |
| 旧 checkpoint  | 无影响（仅数据生成侧改动）                       |
| SFT pipeline   | 无影响（SFT 输出仍为安全代码）                   |
| DPO pipeline   | 影响：垃圾 DPO 对被 skip（而非写入）             |
| Eval pipeline  | 无影响                                           |
| 其他模型       | 兼容（AST 通用、tokenizer API 标准）             |
| CLI 接口       | 不变                                             |
| TRL 兼容       | 兼容（`remove_unused_columns=False` 保留额外列） |

## 验证训练稳定性

1. 运行 `generate_expanded_dataset.py` → 检查 `[DPO] structural skip: 0/N` 输出
2. 运行 `dpo_train.py` → 检查 `DPO tokenization stats` 日志（total/kept/dropped/EOS）
3. 监控前 50 step：`grad_norm < 5` / `logps/chosen` 不低于 -1000 / 无 NaN

## 改动的文件

- `dataset/generate_expanded_dataset.py`
  - 重写 `_extract_likely_param()`（whitelist-first + AST 数据流追踪）
  - 新增 `_validate_dpo_pair_structure()`
  - 修改 `build_dpo_pairs()`（集成校验 + 统计输出）
  - 新增 `_build_dpo_pair_stats()`
