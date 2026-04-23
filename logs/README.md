# 日志目录说明

实验过程与调试信息集中放在此目录，避免写入项目根 `README.md`。

## 建议布局

| 子目录 | 用途 | 格式 |
|--------|------|------|
| `experiments/` | 训练/评测运行时间、参数、样本量、输出路径 | `eval_baseline_20260328T120000Z.log`（纯文本，一行一条或标准 logging） |
| `errors/` | 未捕获异常、Bandit 等调用失败栈 | 同上 |
| `dataset/` | 数据集构建统计、桶分布、去重丢弃条数 | `build_20260328.txt` 或 JSON 一行一条 |

## Python 用法示例

评测时启用文件日志：

```powershell
Set-Location e:\graduation_proj
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model baseline --log-dir logs/experiments
```

或在自定义脚本中调用 `evaluation.experiment_log.setup_file_logging`。
