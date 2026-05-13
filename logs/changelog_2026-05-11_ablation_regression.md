# 消融实验与回归测试框架实现（2026-05-11 第十八次加固）

## 内容

### 1. 消融实验框架 (`scripts/run_ablation.py`)

6 组实验，每组独立 config + 数据生成 + SFT + DPO + 评测：

| Group | 名称             | 变量           | 预期结论         |
| ----- | ---------------- | -------------- | ---------------- |
| A     | Full Pipeline    | 全部组件       | 基线             |
| B     | No DPO           | 跳过 DPO       | DPO 独立贡献     |
| C     | Old Templates    | basic 模板模式 | 模板多样性贡献   |
| D     | No Early Stop    | 禁用早停       | 早停机制贡献     |
| E     | Low Beta DPO     | beta=0.5       | beta=5.0 贡献    |
| F     | Minimal Baseline | C + D 组合     | 全部优化组合贡献 |

用法：
```powershell
# 全部运行
python scripts/run_ablation.py --groups all

# 仅运行 A+B
python scripts/run_ablation.py --groups A B

# 预览命令
python scripts/run_ablation.py --groups all --dry-run

# 从已有结果生成报告
python scripts/run_ablation.py --report
```

### 2. 消融对比分析 (`scripts/compare_ablation.py`)

读取 `outputs/ablation/_all_results.json`，生成：
- `outputs/ablation/ablation_report.md` — Markdown 对比表
- `outputs/ablation/plots/ablation_injection_rate.png` — 注入率对比 bar
- `outputs/ablation/plots/ablation_defense_rate.png` — defense 对比 grouped bar

### 3. 模板模式支持 (`dataset/template_bank.py`)

`TemplateSampler.__init__` 新增 `template_mode` 参数：
- `"full"` (默认): 全部 56 个模板
- `"basic"`: 仅基础类型 (function, try_except, validated) — 模拟旧版 v1

通过环境变量 `ABLATION_TEMPLATE_MODE=basic` 传递给 `generate_expanded_dataset.py`。

### 4. 回归测试套件 (`tests/test_regression_2026_05_11.py`)

18 条测试，5 个测试类：

| 测试类                    | 条数 | 验证内容                                  |
| ------------------------- | ---- | ----------------------------------------- |
| TestTemplateDiversity     | 4    | 模板数≥55、AST有效、重叠<0.70、唯一率≥70% |
| TestDriverDistribution    | 2    | pymysql<40%、无driver>40%                 |
| TestDpoIsomorphism        | 3    | 对≥1500、100%同构、三层存在               |
| TestEvaluationConsistency | 5    | summary字段完整、extraction<10%           |
| TestConfigConsistency     | 3    | beta=5.0、grad_norm=0.3、LR=5e-8          |

运行: `pytest tests/test_regression_2026_05_11.py -v`

## 改动的文件

| 文件                                               | 操作     | 说明                                    |
| -------------------------------------------------- | -------- | --------------------------------------- |
| `scripts/run_ablation.py`                          | **新建** | 消融实验运行器                          |
| `scripts/compare_ablation.py`                      | **新建** | 消融对比分析+绘图                       |
| `configs/ablation/a_full.yaml` ~ `f_minimal.yaml`  | **新建** | 6 组消融实验配置                        |
| `tests/test_regression_2026_05_11.py`              | 已更新   | 修复向后兼容；18条→全部通过             |
| `dataset/template_bank.py`                         | 修改     | TemplateSampler 新增 template_mode 参数 |
| `dataset/generate_expanded_dataset.py`             | 修改     | 读取 ABLATION_TEMPLATE_MODE 环境变量    |
| `logs/changelog_2026-05-11_ablation_regression.md` | **新建** | 本文件                                  |

### 不变更文件
- 所有训练/评测/检测代码 — 零改动
- 现有 configs/default.yaml / dpo.yaml — 不变
