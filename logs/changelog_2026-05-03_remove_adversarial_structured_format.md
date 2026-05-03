# Changelog: Remove adversarial structured format

## 变更范围

- `dataset/adversarial.py`
- `scripts/check_adversarial_dataset.py`
- `README.md`

## 修改内容

### 1) 移除 marker 校验依赖（核心）

- 删除了 `dataset/adversarial.py` 中 marker 结构校验链路（`ADVERSARIAL_MARKERS` 相关依赖路径、marker 完整性检查分支）。
- `check_adversarial_dataset(records)` 语义改为纯 code-only 验证：
  - `output` 必须非空；
  - `ast.parse(output)` 必须通过。

### 2) 简化校验报告结构

- `DatasetCheckReport` 从“对抗结构合规率”改为“语法通过率”：
  - 新字段：`parsed_ok`、`parse_failed`、`parse_pass_rate`。
  - 保留 `violations` 便于定位失败样本。

### 3) CLI 对齐新契约

- `scripts/check_adversarial_dataset.py` 移除对 `ADVERSARIAL_MARKERS` 的导入与打印。
- 输出指标改为 `parsed_ok/parse_failed/parse_pass_rate`。
- CLI 描述改为 code-only 契约，不再描述三段式 marker。

### 4) 文档同步

- README 新增 “Removed adversarial structured format” 更新说明。
- 项目结构中 `adversarial.py` 与 `check_adversarial_dataset.py` 的职责描述同步为 code-only 校验。

## 目的与影响

### 目的

- 消除管线对 `[SECURITY WARNING]/[EXPLANATION]/[SAFE SOLUTION]` 的结构假设。
- 让数据校验与当前训练目标（纯 Python 代码）一致，减少格式耦合和历史兼容负担。

### 影响

- 校验失败将只聚焦于“是否可解析为 Python 代码”，不会再出现 marker 缺失类错误。
- 旧的“结构化对抗输出”合规语义不再作为训练前硬门闸，管线彻底转为 code-only。
