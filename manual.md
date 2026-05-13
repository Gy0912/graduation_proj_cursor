Step 1: 安装依赖
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Required environment

Python venv 已创建（.venv）
训练/评测需要 CUDA GPU（训练脚本有 require_cuda() 硬检查）
bandit 需可调用（检测链依赖）
Explanation 安装项目运行所需依赖（训练、检测、评测、绘图）。

Expected outputs 无固定文件输出（安装日志）。

Step 2: 生成运行配置（default_run）
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe scripts\prepare_default_run.py
Explanation 从 configs/default.yaml 生成 configs/default_run.yaml。
会显式设置：

generation.temperature = 0
eval.enable_taint = true
Expected outputs

configs/default_run.yaml
Step 3: （可选）生成 bandit-only 运行配置
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe scripts\prepare_bandit_only_run.py
Explanation 基于 default_run.yaml 生成 default_bandit_only_run.yaml，并把 outputs 中 JSON 文件名改为 _bandit_only.json 后缀。

Expected outputs

configs/default_bandit_only_run.yaml
Step 4: 生成扩展数据集（核心数据入口）
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42
Explanation 生成训练/评测/DPO 数据，并写 research schema 的 per-task split。
该脚本会调用 write_research_splits(...)，但不会写 eval_fixed.json。

**2026-05-10 第十三次加固**：TemplateSampler 已扩展为 5 维编码风格变异
（5 SQL 查询模式 × 4 结果处理 × 3 import 风格 × 9 注释 × 3 换行 = 1620 组合），
配合 132 函数名 + 50 变量名 + 31 表列组合，确保模板 token 重叠度 <85%。
输出结尾 [DIVERSITY AUDIT] 块自动审计唯一率、Top-1 频率、driver/结构分布。

Expected outputs

data/train_expanded.json
data/eval_expanded.json
data/dpo_pairs.json
data/combined/train.json
data/generation/train.json
data/generation/eval.json
data/fix/train.json
data/fix/eval.json
logs/dataset/generate_expanded_*.log
Step 5: 生成 SFT/DPO demo JSONL（可选但在 README 流程中存在）
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe scripts\build_dataset.py --config configs\default_run.yaml
Explanation 写入 dataset/*.jsonl 的 demo 数据。
注意：该脚本明确不写评测集（不触碰 files.eval_prompts）。

Expected outputs

dataset/sft_train.jsonl
dataset/sft_val.jsonl
dataset/dpo_train.jsonl
Step 6: 构建权威评测集（唯一写入者）
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe scripts\build_eval_fixed.py
Explanation 合并：

data/generation/eval.json
data/fix/eval.json
并做强校验（字段、类型、正负类、写前写后双校验）。

Expected outputs

data/combined/eval_fixed.json
Step 7: （推荐）对抗数据合规复核
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe scripts\check_adversarial_dataset.py
Explanation 独立检查 adversarial 契约（marker 完整、SAFE SOLUTION 参数化、负样本不污染）。

Expected outputs 无新文件（终端 PASS/FAIL 日志）。

训练阶段
训练均从根目录执行；配置按当前代码实际入口使用。
SFT 默认读取 files.train_sft_json（data/combined/train.json）。

Step 8: 训练 LoRA-only（不训练，仅挂载保存）
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe training\train_lora_only.py --config configs\default_run.yaml
Expected outputs

outputs/models/lora_only_starcoder2_3b/
Step 9: 训练 LoRA-SFT
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe training\train_lora_sft.py --config configs\default_run.yaml
Explanation 内部会先跑 run_pretraining_sanity_checks(records)，不通过直接中断训练。

**2026-05-10 第十一次加固**：训练已集成 EarlyStoppingCallback（监控 val_loss + overfit_ratio）。
若低熵数据导致过早过拟合，训练会在 patience=5 步 val_loss 未改善时自动停止，
并将最佳 checkpoint 保存到 best_checkpoint/ 子目录。
DPO 训练入口会自动从最佳 checkpoint 加载（而非最终过拟合 checkpoint）。

Expected outputs

outputs/models/lora_sft_starcoder2_3b/
outputs/models/lora_sft_starcoder2_3b/best_checkpoint/    （若早停触发）
outputs/models/lora_sft_starcoder2_3b/best_checkpoint.json （若早停触发）
Step 10: 训练 LoRA-DPO
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe training\dpo_train.py --config configs\dpo.yaml
Explanation 读取 configs/dpo.yaml，并与 default/default_run 合并配置；依赖 SFT adapter + data/dpo_pairs.json。

**2026-05-10 第十一次加固**：DPO 训练入口通过 resolve_best_sft_checkpoint() 优先从
SFT 输出的 best_checkpoint/ 子目录加载 adapter（若早停触发）；无 best_checkpoint 时
自动回退到最终 checkpoint。

Expected outputs

outputs/models/lora_dpo_starcoder2_3b/
Step 11: 训练 QLoRA-only
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe training\train_qlora_only.py --config configs\default_run.yaml
Expected outputs

outputs/models/qlora_only_starcoder2_3b/
Step 12: 训练 QLoRA-SFT
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe training\train_qlora_sft.py --config configs\default_run.yaml
Explanation 同 Step 9，使用 QLoRA（4bit）量化。集成 EarlyStoppingCallback 早停回调。

Expected outputs

outputs/models/qlora_sft_starcoder2_3b/
outputs/models/qlora_sft_starcoder2_3b/best_checkpoint/    （若早停触发）
outputs/models/qlora_sft_starcoder2_3b/best_checkpoint.json （若早停触发）
Step 13: 训练 QLoRA-DPO
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe training\train_qlora_dpo.py --config configs\dpo.yaml
Explanation QLoRA + DPO，通过 resolve_best_sft_checkpoint 优先从最佳 checkpoint 加载。
评测阶段
Step 14: 评测 baseline
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model baseline
Expected outputs

outputs/baseline_results.json
Explanation 评估基座模型（未经微调）。输出 JSON 的 summary 块中 main 指标层级：

主指标（PRIMARY，方向一致越高越好）：
  defense_success_rate: 对抗 prompt 上输出安全代码的比例（上限 1.0）
  safe_rate_on_benign:  安全 prompt 上输出安全代码的比例（上限 1.0）
辅助指标（AUXILIARY，越低越好）：
  sql_injection_rate_valid: 全局注入率
诊断指标（DIAGNOSTIC）：
  recall_vulnerable / FPR / extraction_failure_rate
Step 15: 评测训练后模型
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model lora_only
.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model lora_sft
.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model lora_dpo
.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model qlora_only
.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model qlora_sft
.\.venv\Scripts\python.exe evaluation\evaluate.py --config configs\default_run.yaml --model qlora_dpo --allow-missing-adapter
Explanation 逐模型生成 *_results.json。
evaluate.py 支持的模型名来自 SUPPORTED_MODELS。

Expected outputs

outputs/lora_only_results.json
outputs/lora_sft_results.json
outputs/lora_dpo_results.json
outputs/qlora_only_results.json
outputs/qlora_sft_results.json
outputs/qlora_dpo_results.json
汇总与结果阶段
Step 16: 汇总比较结果
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe scripts\compare_results.py --config configs\default_run.yaml
Explanation 读取 outputs/*_results.json，输出对比汇总。
当前版本会排除退化基线（lora_only / qlora_only）并检测“完全相同输出”的退化模型。

Expected outputs

outputs/comparison_summary.json
outputs/compare_results.json（若 config 中 outputs.compare_results 存在）
Step 17: 可视化（可选）
Working directory

Set-Location E:\graduation_proj_1
Command

.\.venv\Scripts\python.exe visualization\plot_compare_metrics.py --input outputs\comparison_summary.json --output-dir outputs\plots
.\.venv\Scripts\python.exe scripts\plot_results.py --config configs\default_run.yaml --output-dir outputs\plots
Expected outputs

outputs/plots/injection_rate_valid.png
outputs/plots/extraction_failure_rate.png
outputs/plots/fpr_valid.png
outputs/plots/fnr_valid.png
outputs/plots/safe_rate_valid.png
outputs/plots/f1_conservative.png
outputs/plots/f1_strict.png
outputs/plots/sql_injection_rate_valid.png
outputs/plots/fpr_fnr_valid.png

消融实验
python scripts/run_ablation.py --groups all --dry-run  # 预览
python scripts/run_ablation.py --groups A B C D E F    # 全部运行
python scripts/run_ablation.py --groups A              # 运行A组消融实验

对比分析
python scripts/compare_ablation.py

回归测试
pytest tests/test_regression_2026_05_11.py -v
