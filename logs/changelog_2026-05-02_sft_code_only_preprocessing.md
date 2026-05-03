## 变更标题

SFT 数据集预处理：训练目标规范为仅 SAFE SOLUTION 内纯 Python（code-only）

## 修改内容

- **修改** `training/sft_preprocess.py`
  - 新增 `extract_code_only_completion`：复用 `dataset/adversarial.py::extract_safe_solution`，对含对抗 marker 的样本仅保留 fenced Python 正文；无 marker 时尝试整段或单段 `` ```python `` 抽取；`_collapse_identical_halves` 折叠「整段重复两遍/四遍…」的 completion。
  - 新增 `normalize_sft_records_for_training`：原地重写每条 `output`；`ast.parse` 失败或无法抽取的样本从列表移除，并追加写入 `logs/sft_code_only_dropout.log`（单次运行最多记录 2000 条摘要）。
  - `run_pretraining_sanity_checks` 在原有 `assert_training_outputs_are_python_only` 之前调用规范化；若规范化后无样本则 `RuntimeError`；返回字典增加 `normalize_kept` / `normalize_dropped` / `normalize_drop_log`。
  - 模块文档字符串更新为与「磁盘三段式 vs 训练 code-only」一致。
- **修改** `training/train_lora_sft.py`、`training/train_qlora_sft.py`：仅更新 pre-flight 注释，与实现一致。
- **修改** `README.md`：项目结构树补充 `training/sft_preprocess.py` 等行；执行说明中的 SFT 日志示例；对抗 SFT 章节新增「SFT 训练时的 code-only 投影」；顶栏新增 2026-05-02 变更摘要。

## 变更目的

- 使 **SFT 监督的 completion** 与 **评测中从模型输出抽取并做静态检测的 Python 片段**在语义上对齐（评测侧仍以 `[SAFE SOLUTION]` 优先，见 `detection/sql_injection_detector.py`），减少「训练拟合整段自然语言 + marker，评测只判代码」的分布偏移。
- 去掉 markdown fence 与三段式脚手架后，降低序列长度与 tokenizer 噪声，便于模型专注安全代码模式。

## 影响评估

- **未修改** `evaluation/` 下任何文件；评测契约与对比脚本不变。
- **未修改** 模型定义代码；仅训练数据在内存中的 `records` 被规范化（JSON 源文件默认不自动改写）。
- **数据量**：不符合抽取或语法校验的样本会从本次训练 run 中剔除；需定期查看 `logs/sft_code_only_dropout.log` 审计数据质量。
