# 代码库论文级说明：研究问题、方法体系、数据与评测形式化

**文档定位**：面向学位论文「方法章节 / 系统章节」级别的技术叙述，说明本仓库**研究什么、假设什么、如何实现、如何评测**。**不包含具体实验数值、图表结论或模型排名**；若与源码或 `logs/changelog_*.md` 冲突，以**当前仓库实现**为准。

**配套材料**：复现命令与修复年表见 [`README.md`](README.md)；文件级索引见 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)。

---

## 摘要

本仓库构建了一条面向 **Python SQL 注入（SQL Injection, SQLi）** 的 **代码大语言模型（Code LLM）** 研究管线：以**可控分布的合成数据集**覆盖多种攻击族、难度与任务类型（代码生成 / 漏洞修复），在消费级 GPU 上以 **LoRA / QLoRA** 进行 **监督微调（SFT）**，并可选 **直接偏好优化（DPO）** 强化「安全实现优于同构脆弱变体」的偏好；评测阶段对模型输出执行**确定性代码抽取**，再经**规则层、Bandit（含 B608）与可选动态污点分析**得到逐样本脆弱性判定，并与数据元标签对齐，输出**分层、可解释、对抽取失败鲁棒**的指标集合。工程上强调 **FAIL FAST 数据契约、评测集单一写入者、训练—评测 prompt 对齐、DPO 偏好对与 tokenizer 契约**等，以保证研究可审计、可复现。

---

## 0. 文档范围与非目标

**范围内**：任务定义、数据生成与标注语义、训练目标与优化形式、检测与指标的形式化说明、系统模块职责、可复现流程与工程不变式。

**范围外**：基座模型在客观任务上的绝对能力评价、与工业界未公开系统的对比、对真实 CVE 的完整复现、对法律/合规结论的推断。

---

## 1. 引言与研究动机

### 1.1 背景

SQL 注入长期位居 Web 安全 Top 风险之列。Python 生态中，开发者常通过 `sqlite3`、`pymysql`、`psycopg2`、SQLAlchemy 等与数据库交互；**字符串拼接、错误使用参数化 API、ORM 误用**等模式会引入经典注入面。大模型用于代码补全与生成时，若缺乏针对性约束，可能在对抗或模糊提示下复现上述反模式。

### 1.2 研究问题（本仓库的可操作表述）

1. **安全代码生成（generation）**：给定自然语言指令与输入上下文，模型生成的 Python 模块是否倾向使用**参数化查询或等价安全构造**，并在静态检测视角下尽可能不被判为 SQLi？
2. **安全修复（fix）**：给定含脆弱 SQL 构造的代码片段，模型能否在**保持任务意图**的前提下输出安全实现？
3. **对抗鲁棒性（与元标签对齐）**：当提示被元数据标记为「对抗性」（`expected_vulnerable=True`）时，模型是否仍稳定输出**安全**代码？该问题与「模型输出是否脆弱」的检测信号共同构成主评测维度。
4. **训练recipe 比较**：在同一数据与评测协议下，**仅 LoRA、LoRA+SFT、再接 DPO** 以及 **4bit QLoRA 变体** 等配置的**方法与流程**差异是什么（不涉及具体分差）？

### 1.3 方法与贡献（叙事层面）

- **合成数据**：在攻击类型、难度、任务、`expected_vulnerable`、数据库 driver、代码结构类型等维度上**显式采样**，并通过后验审计（多样性、脆弱模式清零等）约束数据质量。
- **对齐的监督**：磁盘上可出现「对抗三段式」教学文本；进入 SFT 损失的可为 **纯 Python completion（code-only）**，使训练目标与「从模型输出中抽取再检测」的流程一致。
- **偏好优化**：DPO 的 rejected 在同一样本上与 chosen **结构同构**，仅改写 SQL 构造路径，以减少「preference 混入 API 选型噪声」。
- **分层评测**：区分 **抽取成功子集** 与 **全量（含 invalid）** 两种统计口径；引入与「完美安全模型」方向一致的**主指标**，保留传统 P/R/F1 作为**需说明书式解读的诊断量**。

---

## 2. 威胁模型与假设

### 2.1 攻击面

聚焦 **Python 源码层面** 的 SQLi **模式与习惯用法**（拼接、格式化字符串、误用 `text()`、假消毒等），不覆盖存储过程语言、多病种组合漏洞或完整 Web 链路。

### 2.2 对手与提示

数据集将部分样本标注为 **`expected_vulnerable=True`**：表示在该条样本的语境下，**指令或上下文带有诱导脆弱实现的风险或意图**（合成阶段的显式建模）。评测不假设真实攻击者有权修改评测 JSON，但**考查模型在给定静态提示时的输出行为**。

### 2.3 安全保障的含义

「安全」**操作化定义为**：经本仓库实现的**抽取流水线**得到有效 Python 片段后，在配置的检测策略下 **`is_vulnerable=False`**。这是**静态与可选动态启发式判定**，不等价于形式化验证或可利用性证明。

---

## 3. 符号与基本对象

| 符号 | 含义 |
|------|------|
| \(x\) | 单条样本的 **prompt**，由 `training_prompt(instruction, input)` 格式化 |
| \(y\) | 模型输出的 **原始文本**（可能含 markdown、marker、多段文字） |
| \(\mathrm{extract}(y)\) | 评测侧确定性抽取得到的 Python **候选代码**或失败 |
| \(v(\cdot)\) | 检测器给出的 **脆弱性判定**；有效样本上为布尔；抽取失败样本为「未定义」|
| `expected_vulnerable` | 元数据布尔：**提示/场景是否对抗**，**不是**「标准答案应脆弱」 |
| `vulnerability_type` | 攻击族标签（与合成器攻击模板一致） |
| `difficulty` | `easy` / `medium` / `hard` |
| `task_type` | `generation` / `fix` |

**关键语义约定**：SFT 训练目标在「良定义 completion」下为**安全代码**；故在对抗提示上，理想模型仍应产生**不脆弱**输出。这与把「正类」理解为「应检出漏洞」的传统二分类**不同**，第 14 节将形式化其后果。

---

## 4. 系统架构与数据流

### 4.1 逻辑分层

1. **数据合成层**（`dataset/`）：生成扩展 JSON、研究 schema 拆分、DPO 对、多样性审计与对抗校验。
2. **数据治理层**（`scripts/`）：权威评测集合并、独立对抗检查、JSONL demo、结果对比。
3. **训练层**（`training/`）：SFT / DPO、预处理、早停、QLoRA 与 dtype 策略。
4. **推理与评测层**（`evaluation/`）：加载模型与适配器、批量生成、聚合指标、写结构化结果 JSON。
5. **检测层**（`detection/`）：代码抽取、规则、Bandit、可选污点。

### 4.2 端到端数据流（示意）

```mermaid
flowchart LR
  subgraph synth [合成]
    G[generate_expanded_dataset.py]
    TB[template_bank.py]
    ADV[adversarial.py]
    G --> TB
    G --> ADV
  end
  subgraph artifacts [数据产物]
    TR[combined/train.json]
    GE[generation/eval.json]
    FE[fix/eval.json]
    DPO[dpo_pairs.json]
  end
  G --> TR
  G --> GE
  G --> FE
  G --> DPO
  subgraph eval_build [评测集治理]
    BEF[build_eval_fixed.py]
  end
  GE --> BEF
  FE --> BEF
  BEF --> EVF[eval_fixed.json]
  subgraph train [训练]
    SFT[train_*_sft.py]
    DPOT[dpo_train.py]
    PP[sft_preprocess.py]
  end
  TR --> PP --> SFT
  SFT --> DPOT
  DPO --> DPOT
  subgraph ev [评测]
    EVAL[evaluate.py]
    DET[sql_injection_detector.py]
    MET[metrics.py]
  end
  EVF --> EVAL
  EVAL --> DET --> MET
```

---

## 5. 数据集：Schema、采样与标签

### 5.1 扁平扩展集（生成器直接输出）

`dataset/generate_expanded_dataset.py` 文档头定义每条样本携带（概念上）：`instruction`、`input`、`output`、`attack_type`、`difficulty`、`task_type`、`expected_vulnerable`、`schema_table`、`schema_column` 等。其中 **`output` 在磁盘上始终为安全 Python**（经 `ast.parse` 与脆弱模式扫描），无法通过校验的候选在生成循环中丢弃并重试。

### 5.2 研究 Schema（训练与 per-task 拆分）

`dataset/research_schema.py` 将样本规范为含下列核心字段的记录（训练含 `output`，评测行可不含）：

- `id`：字符串。若缺失则 `stable_sample_id` 由 `task_type`、`vulnerability_type`（或 `attack_type`）、`instruction`、`input` 的哈希导出，前缀 `sqlsec-`。
- `task_type`、`instruction`、`input_code`
- `expected_vulnerable`：**必须显式存在且为 `bool`**（训练与评测均禁止默认回退）
- `vulnerability_type`、`difficulty`

**写入路径**：

- `data/combined/train.json`：全任务训练集。
- `data/generation/{train,eval}.json`、`data/fix/{train,eval}.json`：按任务拆分。

**刻意不写入**：`data/combined/eval_fixed.json`（由 `scripts/build_eval_fixed.py` 独占写入，见 §5.4）。

### 5.3 分布设计（概念参数）

生成器内定义**非均匀**权重（具体数值以源码为准）：

- **攻击类型** `ATTACK_TYPES`：含 `string_concat`、`fstring`、`format_string`、`fake_sanitization`、`orm_misuse`、`parameterized_query`、`indirect_injection` 等；权重向易混淆与间接注入倾斜。
- **难度**：训练 `easy/medium/hard` 与评测 **hard 占比更高**（评测更难）。
- **任务**：`generation` 与 `fix` 约各半。
- **`expected_vulnerable`**：目标比例约 **0.5**，用于保证混淆矩阵两侧可辨。

样本量由 CLI（如 `--num_samples`、`--eval_ratio`、`--seed`）控制，支持复现实验。

### 5.4 权威评测集 `eval_fixed.json`

`scripts/build_eval_fixed.py`：

- **输入**：`data/generation/eval.json` + `data/fix/eval.json`。
- **操作**：schema 强校验、`id` 去重、**正类/负类均非空**。
- **输出**：`data/combined/eval_fixed.json`；**写入前与写回磁盘后各做一次**完整性断言。

该设计消除「多脚本写同一评测文件导致标签漂移」的风险，是论文方法中**可复现评测协议**的关键环节。

---

## 6. 安全模板库与结构多样性（`dataset/template_bank.py`）

### 6.1 设计目标

缓解 **低熵记忆**：若安全参考实现仅来自少数骨架，模型会快速压低训练 loss，但对训练外 token 分配极端概率，进而导致后续 DPO 等阶段数值不稳定。**模板库 v2** 以 **AST 级异构模板** 为主：每个模板为自包含、可解析、不含脆弱模式的参数化代码；在**控制流、错误处理、API 选型、异步/装饰器/类结构**等多维上与多数其他模板区分。

### 6.2 Driver 与结构类型

- **Driver**（示例）：`pymysql`、`sqlite3`、`sqlalchemy`、`psycopg2`、`mysql-connector`、`aiomysql`、`asyncpg`（见 `ALL_DRIVERS`）。
- **结构标签**（示例）：`function`、`class`、`context_manager`、`decorator`、`async`、`orm`、`generator`、`closure`、`try_except`、`validated`、`multi_func`、`dispatch` 等。

### 6.3 重要性采样（直观表述）

`DRIVER_TARGET_WEIGHTS` 等给出**目标边际分布**；`TemplateSampler` 结合历史计数做**重要性采样**，使低频模板获得更高被抽中概率，从而抬高 **唯一输出率** 与 **driver/结构覆盖**，并在 `generate_expanded_dataset.py` 收尾打印 **DIVERSITY AUDIT**（具体统计项以日志为准）。

### 6.4 与检测侧的关系

模板库**不 import** 检测模块；安全性的硬约束在生成器侧通过 `adversarial.contains_vulnerable_sql_pattern` 与 `ast.parse` 等保证。规则层演化时，changelog 指导是否同步副本（`adversarial.py` 内 **vendored** 正则集）。

---

## 7. 对抗格式化、`adversarial.py` 与 code-only 监督

### 7.1 动机

若将**可执行脆弱 SQL** 直接写入 SFT 的 `output`，则在 next-token 交叉熵下，模型被显式优化的目标包含「生成注入构造」——属于**训练数据污染**。本仓库改为：对需体现对抗语义的数据，`output` 可为 **警告 + 解释 + 安全实现**三段结构；其中可执行代码段必须参数化并由机器可检。

### 7.2 模块职责摘录

- **常量**：`MARKER_WARNING`、`MARKER_EXPLANATION`、`MARKER_SAFE`，与评测 `metrics.py` / `evaluator.py` **字面一致**。
- **`build_secure_response`**：按攻击族生成三段式文本，SAFE 段为安全代码。
- **`extract_code_only_completion`**：从原始 `output` 抽取 **第一段** ```python fenced** 或可 `ast.parse` 的全文；含折叠整块重复的实现。
- **`contains_vulnerable_sql_pattern`**：对代码扫描已命名脆弱模式列表（拼接、`text(f"...")`、`execute(...+...)`、`".join"` 等），返回命中名列表。

### 7.3 训练路径：`training/sft_preprocess.py`

流程要点：

1. **`normalize_sft_records_for_training`**：将每条记录的 `output` **替换为**抽取后的纯 Python；失败样本丢弃并记入 `logs/sft_code_only_dropout.log`。
2. **`FORBIDDEN_TRAINING_TOKENS`**：禁止 completion 再次出现三段式 marker 与已识别的数据泄漏占位（如 `# ref=`），以避免模型学会输出「看起来像对抗脚手架」却无安全语义。
3. **`run_pretraining_sanity_checks`**：训练前断言对抗合规与「输出不含脆弱模式」；与 `scripts/check_adversarial_dataset.py` 形成外层/内层交叉验证。

**论文叙述建议**：将这种安排描述为 **「磁盘级教学 rich target」与「计算图级纯代码 LM 目标」的显式分裂**，有利于与静态检测流水线对齐。

---

## 8. DPO：偏好数据、同构性与稳定训练

### 8.1 学习目标（概念）

DPO（Rafailov et al.；实现基于 Hugging Face TRL）在策略 \(\pi_\theta\) 与参考 \(\pi_{\mathrm{ref}}\) 之间，用配对 \((z; y^+, y^-)\) 优化：相对提高 **`chosen`** 似然、降低 **`rejected`** 似然，并含隐式 KL 约束（由 \(\beta\) 控制强度）。本仓库中 \(z\) 为 **prompt** 字段。

### 8.2 数据构造原则

- **只对 `expected_vulnerable=True` 的样本建对**：良性提示下 SFT 已足以学到安全输出，再加入「安全优于脆弱」信号易冗余。
- **`chosen`**：安全实现（常与 SFT 目标一致）；经 code-only 规范化与语法校验。
- **`rejected`**：在同一样本的 instruction/schema 语境下对 SQL 片段做**脆弱化**；优先 **AST 级局部手术**（保留 import、函数签名、变量名与非 SQL 逻辑），失败则走 **与 chosen 对齐的生成回退**，避免「pymysql 安全 vs sqlalchemy 脆弱」式假偏好。
- **fix 任务**：历史问题包括「input 脆弱代码范式」与「chosen 全新安全范本范式」错位；现为从脆弱 input 抽取语境，**同 driver 参数化** 得 chosen，保证偏好落在「修复策略」而非「换库」。

### 8.3 Tokenization 与数值稳定（`training/stable_dpo_trainer.py`）

- **BOS 边界**：prompt 使用 `add_special_tokens=True`；chosen/rejected 片段避免重复插入特殊 token，以贴近推理时序列布局。
- **EOS 与 completion 边界**：实现遵循 TRL 惯例——对 completion 追加 EOS 后，以 **合并 tokenize 再拆分** 得到 `prompt_ids` 与 `chosen_ids`/`rejected_ids`，供 `DPODataCollator` 构建 mask。**禁止**早期版本中手工拼接 id 导致的掩码错位。
- **`dataset_num_proc` 强制为 1**：规避多进程与 Accelerate PartialState 的已知崩溃路径。
- **防护**：正向 hook clamp logits；`DpoNanGuardCallback` 在优化步前检测 NaN 梯度。

### 8.4 配置文件中的超参语义（不涉及取值优劣）

`configs/default.yaml` 中 `dpo.beta` 控制偏好强度与 KL 惩罚折衷；`max_length` / `max_prompt_length` 与截断共生——过短 prompt 若使 chosen/rejected 预算为零，样本应被丢弃（见对应 changelog），否则产生零梯度步。

---

## 9. 模型与参数高效微调

### 9.1 基座

默认 **`bigcode/starcoder2-3b`**：因果语言模型，面向代码。配置允许改为更小模型以适配显存。

### 9.2 LoRA（Low-Rank Adaptation）

在选定线性层上注入低秩分解 \(\Delta W = B A\)，冻结基座大部分参数。`training/lora_utils.py` 支持 `lora_target_modules: auto`，对 StarCoder2 结构匹配 `q_proj/k_proj/v_proj/o_proj/c_fc/c_proj` 等（以实际匹配结果为准）。

### 9.3 QLoRA

训练脚本可选 **4bit 量化基座 + LoRA**，在 `BitsAndBytesConfig` 下加载；与 `dtype_utils`、是否禁用 AMP 等共同决定数值路径。默认配置面向约 **8GB 显存** 可训练可推理的典型场景。

### 9.4 SFT 目标函数（概念）

对每条样本，令 \(x\) 为 prompt token 序列，\(c\) 为 completion token 序列。训练使用 **completion-only loss**：

\[
\mathcal{L}_{\mathrm{SFT}} = - \sum_{t} \mathbf{1}[t \in \text{completion}] \log p_\theta(c_t \mid x, c_{<t})
\]

prompt 段不贡献梯度。实现上由 `completion_mask` 与 TRL/DataCollator 协同完成。

---

## 10. 训练管线衔接

### 10.1 数据路径 Fail-Fast

`train_lora_sft.py` / `train_qlora_sft.py` 在配置缺失或 `train_sft_json` 不存在时**立即失败**，禁止静默回退到小型旧 JSON，以避免 schema 与 code-only 契约被破坏。

### 10.2 早停与 DPO 起点（`training/early_stopping.py`）

- **`EarlyStoppingCallback`**：按 `val_loss` 改善情况与 `patience` / `min_delta` 早停；可选监控 `overfit_ratio = train_loss / val_loss` 并发出过拟合警告。
- **最佳 checkpoint**：在 `best_checkpoint/` 另存，`best_checkpoint.json` 记录 marker。
- **`resolve_best_sft_checkpoint`**：`dpo_train.py` 优先从最佳点加载 SFT 适配器，避免「最终 step 过拟合」进入 DPO。

---

## 11. 推理与解码

### 11.1 默认解码

配置中常见 **`temperature=0`（贪心）** 与 `top_p` 等场合适用参数。贪心下存在 **重复 token 循环**风险，故引入 **`repetition_penalty>1`**（温和）及可选 `no_repeat_ngram_size`（默认 0 以避免过度破坏代码结构）。

### 11.2 EOS 截断

解码后在 **首个 EOS** 处截断 token 再转文本，避免模型在未自然结束时输出第二段残缺代码块，导致抽取器误取末段碎片。

### 11.3 `code_only_training` 评测旗标

当 SFT 仅训练纯代码时，设置 `eval.code_only_training=true`：**响应结构 marker 命中率**在解释上应接近 0；指标仍计算以便审计，但不应解读为「模型忘记安全」。

---

## 12. 漏洞检测子系统（`detection/`）

### 12.1 规则层（`rule_based.py`）

基于正则/启发式的模式集合，覆盖典型字符串构造与 API 误用。历史上移除过将 **合法参数化** `execute("...%s", (x,))` 误判为 `%` 格式化漏洞的规则；测试 `tests/test_rule_false_positive.py` 防止回归。

### 12.2 Bandit（`bandit_wrapper.py`）

子进程运行 Bandit JSON 报告。默认合并策略在 `or` 模式下强调 **B608** 作为 SQL 构造相关信号（具体合并逻辑以 `sql_injection_detector.py` 为准）。

### 12.3 污点追踪（可选）

`detection/taint_tracker.py` 提供动态路径；评测通过 CLI `--enable-taint` 打开。计算开销通常高于纯静态。

### 12.4 合并模式 `merge_mode`

- **`or`**：规则或（限定）Bandit 信号或污点命中即判脆弱（实现细节见源码）。
- **`or_bandit_any`**：任意 Bandit 问题也可触发（更敏感）。
- **`weighted`**：对多通道打分并与阈值比较。

---

## 13. 代码抽取规范（评测侧）

评测将模型输出视为**半结构化文本**，`detection/sql_injection_detector.py` 中逻辑可概括为：

1. 若存在 `[SAFE SOLUTION]`：**仅**允许该段提供代码，且须为带 `python` 标记的 fenced 块并通过 `ast.parse`。
2. 若无 `[SAFE SOLUTION]`：接受 ```python fenced 块。
3. 若无 fenced：仅当**全文**为合法 Python 时接受。

**预处理**：去孤立 fence 行、折叠重复输出、失败时逐行回退尝试解析等，与数据集侧行为**刻意对齐**以减少「训练能 parse、评测不能 parse」的鸿沟。

**失败样本**：`invalid_extraction=True`，并记录 debug 信息至 `outputs/debug_invalid_samples.json`（上限与字段以 evaluator 为准）。

---

## 14. 评测指标：形式化与层级

### 14.1 抽取状态

令 \(I\) 为 invalid 指示量。若 \(I=1\)，则 **\(v\) 未定义**；实现上 `is_vulnerable=None`。

### 14.2 检测器输出上的有效子集

在 \(I=0\) 的 **valid** 子集上定义：

- **SQL 注入率（valid）**：\(\hat{p}_{\mathrm{inj}} = \frac{1}{n_{\mathrm{valid}}} \sum \mathbf{1}[v=\text{true}]\)。
- **安全率（valid）**：\(1 - \hat{p}_{\mathrm{inj}}\)。

### 14.3 主指标（方向一致，2026-05-10 层级）

令 \(\mathcal{P}_+=\{i: \texttt{expected\_vulnerable}_i=\text{true}\}\)，\(\mathcal{P}_-=\{i: \texttt{expected\_vulnerable}_i=\text{false}\}\)。在 **valid 且可判定** 前提下：

- **defense_success_rate**：\(\Pr(v=\text{false} \mid i\in\mathcal{P}_+)\)。**越高越好**。
- **safe_rate_on_benign**：\(\Pr(v=\text{false} \mid i\in\mathcal{P}_-)\)。**越高越好**。

「完美恒安全模型」在二者上均趋于 1，而传统 **recall\_vulnerable** 会趋于 0——这**不是矛盾**，而是标注语义与「正类」定义导致的**可预期的表观现象**。

### 14.4 `valid_only_metrics` 中的分类表

`metrics.py` 将 `expected_vulnerable` 作为 **y_true**，`is_vulnerable` 作为 **y_pred**，在 valid 上统计 TP/FP/TN/FN 并给出 precision/recall/F1/FPR/FNR。**注意**：在此编码下，**更高的 recall\_vulnerable 表示模型更常在对抗提示下产出脆弱代码，即更不安全**——论文中必须配文字说明，或仅作诊断引用。

### 14.5 全量样本：conservative vs strict

对 \(I=1\)：

- **conservative**：若 `expected_vulnerable=True` 记 **FN**；若 `False` 记 **TN**。含义：抽取失败不奖励为「安全」。
- **strict**：若 `expected_vulnerable=True` 记 **FN**；若 `False` 记 **FP**。含义：抽取失败在良性侧也惩罚，**关闭「靠乱码逃避 FP」的漏洞**。

### 14.6 抽取失败硬阈值

若 `extraction_failure_rate = n_invalid / n_samples > 0.5`，`aggregate_metrics` **拒绝**写出评测 JSON，防止「大部分输出不可解析」仍产生看似完整的指标表。

### 14.7 响应质量（与脆弱性正交）

对 `raw_output` 做子串匹配：`has_warning`、`has_explanation`、`has_safe_solution`，进而在**全样本**上统计 `warning_rate`、`explanation_rate`、`safe_solution_rate`、`full_compliance_rate` 及按 `expected_vulnerable` 拆分的子集率。该组指标**不替代**安全检测，刻画**结构化教学输出**是否出现。

### 14.8 分层与辅助指标

`explain_metrics()` 给出完整文字版层级：**PRIMARY**（defense / benign）、**AUXILIARY**（如 valid 注入率、抽取失败率、full_compliance 在 code-only 下的解释）、**DIAGNOSTIC**（recall、FPR 等）。论文建议以 **PRIMARY** 为摘要主表，**DIAGNOSTIC** 放附录并强制脚注语义。

---

## 15. 实验因素与对照设计空间（无数值）

可在方法章节列成**因素表**（具体水平以配置为准）：

| 因素 | 水平示例 |
|------|---------|
| 适配器 recipe | baseline / lora_only / lora_sft / lora_dpo / qlora_* |
| 检测合并 | `or` / `or_bandit_any` / `weighted` |
| 污点 | 开 / 关 |
| 训练数据规模 | `--num_samples`、eval 比例、seed |
| code-only 训练 | `code_only_training` true/false 对响应质量解释的影响 |
| DPO | 有 / 无；β、学习率、是否从 best SFT 启动 |

---

## 16. 可复现性、工程不变式与测试

### 16.1 关键不变式（建议论文「威胁效度」或「工程保障」一小节）

1. **评测集单一写入者**与双重 schema 校验。
2. **`expected_vulnerable` 显式 bool**，全管线禁止默认 False。
3. **invalid 样本 `is_vulnerable=None`**，禁止混入「全样本安全率」。
4. **训练—评测 prompt 模板统一**（`Instruction:\n...\n\nInput:\n...\n\n`）。
5. **DPO 偏好同构**与结构化校验（skipped 计数可审计）。
6. **changelog + 针对性单元测试**锁死历史上的语义漏洞（regex 假阳性、invalid 指标等）。

### 16.2 测试资产

`tests/` 含 invalid extraction、响应质量、对比脚本、规则假阳性、污点等用例；论文明示「回归测试覆盖的属性」可增强可信度。

---

## 17. 局限性

1. **检测是启发式合集**：既不能保证无漏报，也不能保证无误报；规则与 Bandit 更新会改变评测可比性。
2. **合成数据分布 ≠ 真实仓库分布**：外部有效性需审慎表述。
3. **SQLAlchemy / 异步等覆盖受模板与生成器约束**，不代表所有 ORM 误用变种。
4. **8GB 配置**偏重资源可行，可能与大规模训练最佳实践不同。

---

## 18. 可复现操作索引（命令级）

下列命令应与 `README.md` 完全一致；此处仅作论文章节交叉引用：

1. 安装依赖：`pip install -r requirements.txt`
2. 生成配置：`python scripts/prepare_default_run.py`
3. 合成数据：`python dataset/generate_expanded_dataset.py --num_samples <N> --eval_ratio <r> --seed <s>`
4. （可选）`python scripts/build_dataset.py --config configs/default_run.yaml`
5. 合并评测：`python scripts/build_eval_fixed.py`
6. （可选）`python scripts/check_adversarial_dataset.py`
7. 训练：`python training/train_lora_sft.py --config configs/default_run.yaml` 等
8. DPO：`python training/dpo_train.py --config configs/dpo.yaml`
9. 评测：`python evaluation/evaluate.py --config configs/default_run.yaml --model <name>`
10. 对比：`python scripts/compare_results.py`（参数以脚本为准）

---

## 附录 A：配置键语义（节选）

| 配置路径 | 语义 |
|-----------|------|
| `model.base_model` | Hugging Face 模型 id 或路径 |
| `files.train_sft_json` | SFT JSON 数据源 |
| `files.eval_prompts` | **权威评测集路径** |
| `files.dpo_pairs` | DPO JSONL/JSON 路径 |
| `generation.*` | `max_new_tokens`、`temperature`、`repetition_penalty` 等 |
| `eval.merge_mode`、`eval.enable_taint`、`eval.code_only_training` | 检测与语义旗标 |
| `training.*` | 学习率、epoch、LoRA rank、梯度裁剪、早停参数等 |
| `dpo.beta`、`dpo.max_length` | DPO 特有 |

---

## 附录 B：主要产物路径

| 产物 | 典型路径 |
|------|-----------|
| 训练适配器目录 | `outputs/models/lora_sft_starcoder2_3b/` 等 |
| 单次评测输出 | `outputs/<recipe>_results.json` |
| 跨配方对比 | `outputs/comparison_summary.json` |
| 数据集日志 | `logs/dataset/generate_expanded_*.log` |

---

## 参考文献（占位，按需补全）

1. Raffel et al. / Vaswani et al.（Transformer）；  
2. Hu et al., **LoRA**；  
3. Dettmers et al., **QLoRA**（若正文使用 4bit 训练叙事）；  
4. Rafailov et al., **DPO**；  
5. Hugging Face **TRL** `DPOTrainer` 文档；  
6. Bandit **B608** 规则说明。

---

*修订说明：本文为 `CODEBASE_DETAILED_OVERVIEW.md` 的论文级扩写版本，用于方法与系统描述；不包含实验结果解读。*
