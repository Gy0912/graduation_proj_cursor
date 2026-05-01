# Changelog 2026-05-01: Training Data Python-Only Targets

## 修改文件
- `dataset/generate_expanded_dataset.py`
- `training/sft_preprocess.py`

## 变更内容
- 训练样本 `output` 改为仅保留 Python 代码，不再包裹结构化响应模板。
- 移除训练数据生成中的模板化输出路径（`[SECURITY WARNING]` / `[EXPLANATION]` / `[SAFE SOLUTION]`）。
- 移除 DPO 训练 prompt 中的 `### Response` 分节模板，改为纯文本 `Instruction + Input`。
- 更新 SFT 预处理默认模板，移除 `### Response` 分节。
- 新增训练前硬校验：`output` 必须是纯 Python（`ast.parse` 可通过），且不得含以下模板 token：
  - `[SECURITY WARNING]`
  - `[EXPLANATION]`
  - `[SAFE SOLUTION]`
  - `### Response`
  - `### Solution`
  - `### Test`

## 目的与影响
- 目的：清除训练目标中的结构化自然语言模板，避免模型学习到非代码格式输出。
- 影响：SFT/DPO 的训练目标统一为“纯 Python 代码”，与代码提取/语法可解析目标一致。
- 约束：本次未修改 evaluation 逻辑与 extractor 逻辑。
