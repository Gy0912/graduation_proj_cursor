## 变更标题

评测推理链路：code-only 提示、token 上限、早停与 decode 后清洗

## 修改内容

- **新增** `evaluation/inference_constraints.py`
  - `CODE_ONLY_INSTRUCTION_SUFFIX` / `append_code_only_instruction`：追加「仅输出合法 Python、禁止解释与 markdown」英文指令。
  - `clamp_max_new_tokens`：将配置中的 `max_new_tokens` 与 `inference_max_new_tokens_cap` 取 min。
  - `build_stopping_criteria`：仅在 **batch=1** 时返回 `StoppingCriteriaList`（生成后缀出现 `\n\n` 或尾部半段重复时早停）；多 batch 时返回 `None`（HF 整批同停语义限制）。
  - `postprocess_generated_text`：去行首编号/散文、按双空行取最短可 `ast.parse` 前缀、去 fence、调用 `extract_code_only_completion`、折叠连续重复行。
- **修改** `evaluation/evaluator.py` 中 `run_eval_on_prompts`：**仅**生成与 decode 路径——对 encode 用追加后缀的 prompt、`max_new_tokens` 使用裁剪值、传入 `stopping_criteria`、decode 后可选 `postprocess_generated_text`。**未改** `_per_sample_from_detection` / `aggregate_metrics` / invalid 样本契约。
- **修改** `evaluation/evaluate.py`：从 `generation` 读取 `code_only_inference`、`inference_max_new_tokens_cap` 传入 `run_eval_on_prompts`；结果 `meta` 记录上述开关。
- **修改** `configs/default.yaml`、`configs/default_run.yaml`、`configs/default_bandit_only_run.yaml`：增加 `code_only_inference` 与 `inference_max_new_tokens_cap`。
- **修改** `README.md`：项目树与 §6 评测说明文档化推理约束。

## 变更目的

- 缓解推理时编号前缀、重复代码、解释性文字与 markdown 围栏导致的抽取失败与指标噪声。
- 与数据侧 `extract_code_only_completion` 语义对齐，使 `raw_output` 更接近可解析 Python。

## 影响评估

- **训练代码**：未修改 `training/`。
- **评测指标定义**：未改 `evaluation/metrics.py`；`raw_output` 仍为模型侧文本经清洗后的版本，结构合规子串匹配行为可能随模型输出变短而变化，属预期 trade-off。
- **关闭开关**：YAML 中 `code_only_inference: false` 可恢复近似旧推理行为（仍受配置 `max_new_tokens` 约束）。
