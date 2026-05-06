"""从评测文件加载评测样本（自动识别 JSON 数组或 JSONL）。

严格标签契约（修复 missing-label 评测 Bug 后）:
  - 每条样本必须显式包含 "expected_vulnerable" 键，且其值为 Python bool。
  - 缺失或类型不匹配时，立即抛出 ValueError（FAIL FAST）。
  - 不提供任何默认值回退；禁止 `row.get("expected_vulnerable", False)` 等静默模式。
  - 同时强制样本具备可构造 prompt 的最小字段（prompt 或 instruction+input）。

2026-04-20 四次加固扩展：
  - 对外暴露 ``validate_eval_samples(samples)``：用于在加载后做一次独立的"pre-eval
    validation step"，承接 [TASK 3]「Before evaluation: assert all(...)」。
  - 返回结构化统计 ``{"total", "valid", "pct_valid", "pos", "neg"}``，供 CLI
    打印 [TASK 4] 需要的"% samples with valid labels"日志。

此模块被 `evaluation/evaluate.py` 直接调用；被强校验拒绝的样本会中断整次评测，
避免像旧流程那样把一整批无标签样本送进分类指标并得到 TP=FN=0 的假象。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CRITICAL_SAMPLE_FIELDS: tuple[str, ...] = ("id", "prompt", "expected_vulnerable")


def _instruction_input_prompt(instruction: str, user_input: str) -> str:
    """回退提示构造：与训练端 ``training_prompt()`` 格式一致。

    2026-05-05 修复（问题 #12）：旧版使用 "Generate one Python module..."
    / "Do not output markdown, code fences..." 等指令——与 SFT 训练分布
    （``Instruction:\\n...\\n\\nInput:\\n...\\n\\n``）完全不一致，
    且「no fences」指令在训练数据中不存在。现改为与
    ``dataset/generate_expanded_dataset.py::training_prompt()`` 相同格式。
    """
    return (
        "Instruction:\n"
        + (instruction or "").strip()
        + "\n\nInput:\n"
        + (user_input or "").strip()
        + "\n\n"
    )


def load_eval_prompts(path: Path) -> list[dict]:
    """
    支持两种格式（由首个非空白字符判断）：
    - 以 ``[`` 开头：标准 JSON 数组
    - 否则：JSONL（每行一个 JSON 对象）

    注意：无论来源是哪种格式，所有样本都会经过 `_normalize_sample` 的强 schema 校验。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")

    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"评测文件为空: {path}")

    stripped = text.lstrip()
    first_char = stripped[0]

    if first_char == "[":
        print("Detected JSON format")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"评测 JSON 解析失败 ({path}): {e.msg} (line {e.lineno}, column {e.colno})"
            ) from e
        if not isinstance(data, list):
            raise ValueError(f"{path} 应为 JSON 数组（顶层必须是列表）")
        samples: list[dict] = []
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path} 数组第 {idx} 个元素必须是 JSON 对象，"
                    f"得到 {type(row).__name__}（无标签元素一律拒绝）"
                )
            try:
                samples.append(_normalize_sample(row))
            except ValueError as e:
                raise ValueError(f"{path} 数组第 {idx} 条样本无效: {e}") from e
        return samples

    print("Detected JSONL format")
    samples = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        try:
            row = json.loads(line_stripped)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"评测 JSONL 第 {line_no} 行解析失败 ({path}): "
                f"{e.msg} (column {e.colno})"
            ) from e
        if not isinstance(row, dict):
            raise ValueError(
                f"评测 JSONL 第 {line_no} 行应为 JSON 对象，"
                f"得到 {type(row).__name__}"
            )
        try:
            samples.append(_normalize_sample(row))
        except ValueError as e:
            raise ValueError(f"评测 JSONL 第 {line_no} 行样本无效: {e}") from e
    return samples


def _require_bool(row: dict[str, Any], key: str) -> bool:
    """严格校验键存在且值为 Python bool（不容忍 0/1/'True' 等近似类型）。"""
    if key not in row:
        raise ValueError(
            f"Missing {key} in evaluation sample (id={row.get('id')!r}). "
            "评测样本必须显式包含此标签，禁止任何默认值回退。"
        )
    value = row[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"{key} 必须是 bool 类型，实际为 {type(value).__name__}: {value!r} "
            f"(id={row.get('id')!r})"
        )
    return value


def _require_nonempty_str(row: dict[str, Any], key: str) -> str:
    """严格校验键存在且值为非空字符串。"""
    if key not in row:
        raise ValueError(
            f"Missing {key} in evaluation sample (id={row.get('id')!r})."
        )
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{key} 必须是非空字符串，实际为 {value!r} (id={row.get('id')!r})"
        )
    return value


def _normalize_sample(row: dict[str, Any]) -> dict[str, Any]:
    """
    将原始样本字典标准化为评测器所需的字段集合。

    **强制契约**（FAIL FAST，任一不满足即 raise ValueError）:
      1) 必须显式包含 "expected_vulnerable"，且类型为 bool。
      2) 必须能构造非空 prompt：显式 "prompt" 非空，或 "instruction"/"input" 组合非空。
      3) 必须显式包含 "vulnerability_type"（或兼容的 "attack_type"）与
         "difficulty"、"task_type"，且为非空字符串。
      4) 必须显式包含非空字符串 "id"。id 是不透明哈希形式（``sqlsec-<hex20>``），
         **不允许**做任何数字解释；这里在加载入口就卡住，让下游永远看不到数字 id
         或 None。
    """
    sample_id_raw = row.get("id")
    if not isinstance(sample_id_raw, str) or not sample_id_raw.strip():
        raise ValueError(
            f"id 必须是非空字符串（不透明哈希形式，如 'sqlsec-<hex>'），"
            f"实际为 {type(sample_id_raw).__name__}: {sample_id_raw!r}"
        )

    expected_vulnerable = _require_bool(row, "expected_vulnerable")

    prompt = row.get("prompt")
    if not (isinstance(prompt, str) and prompt.strip()):
        instruction = row.get("instruction", "")
        user_input = row.get("input")
        if user_input is None:
            user_input = row.get("input_code", "")
        user_input_str = "" if user_input is None else str(user_input)
        if not (isinstance(instruction, str) and instruction.strip()) and not user_input_str.strip():
            raise ValueError(
                f"无法构造 prompt：prompt/instruction/input 均为空 "
                f"(id={row.get('id')!r})"
            )
        prompt = _instruction_input_prompt(str(instruction or ""), user_input_str)

    vuln_type = row.get("vulnerability_type") or row.get("attack_type")
    if not (isinstance(vuln_type, str) and vuln_type.strip()):
        raise ValueError(
            f"vulnerability_type 必须是非空字符串，实际为 {vuln_type!r} "
            f"(id={row.get('id')!r})"
        )

    difficulty = _require_nonempty_str(row, "difficulty")
    task_type = _require_nonempty_str(row, "task_type")

    return {
        "id": sample_id_raw,
        "prompt": prompt,
        "instruction": row.get("instruction"),
        "input": row.get("input") if row.get("input") is not None else row.get("input_code"),
        "input_code": row.get("input_code"),
        "output": row.get("output"),
        "attack_type": vuln_type,
        "vulnerability_type": vuln_type,
        "difficulty": difficulty,
        "task_type": task_type,
        "expected_vulnerable": expected_vulnerable,
    }


def validate_eval_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    独立的 pre-eval validator（承接 [TASK 3]「assert all("expected_vulnerable" in s ...)」）。

    与 ``evaluator._assert_dataset_sanity`` 的区别：

      - 本函数**只做 loader-level 校验**（critical field 存在性 + 基本类型），
        不依赖 detector 输出字段；可以在 ``evaluate.py`` 的最早入口（模型加载之前）
        调用，省去一次 GPU 初始化。
      - ``evaluator._assert_dataset_sanity`` 在推理管线正式启动时重复校验，并加入
        正负双类检查，双重保险。

    严格规则：对每条 sample，必须同时：
      1. 包含 ``id``、``prompt``、``expected_vulnerable`` 三个 critical field；
      2. ``id`` / ``prompt`` 为非空字符串；
      3. ``expected_vulnerable`` 为 Python ``bool``。

    任一违规均 raise ``RuntimeError``，错误信息定位到具体 idx + id + 字段 + 原因。

    返回统计 dict（用于上层打印 [TASK 4] 的百分比日志）：

        {
          "total": int,         # 样本总数
          "valid": int,         # 通过强校验的样本数（== total，否则 raise）
          "pct_valid": float,   # 百分比（== 100.0，除非后续加入 soft-mode）
          "pos": int,           # expected_vulnerable == True 的样本数
          "neg": int,           # expected_vulnerable == False 的样本数
        }
    """
    if not isinstance(samples, list):
        raise RuntimeError(
            f"validate_eval_samples: samples 必须是 list，实际为 {type(samples).__name__}"
        )
    if not samples:
        raise RuntimeError(
            "validate_eval_samples: 样本列表为空，无法启动评测。"
        )

    violations: list[tuple[int, Any, str, str]] = []
    for idx, s in enumerate(samples):
        if not isinstance(s, dict):
            violations.append(
                (idx, None, "<sample>",
                 f"not a dict, got {type(s).__name__}: {s!r}")
            )
            continue

        sid_val = s.get("id")
        for fld in CRITICAL_SAMPLE_FIELDS:
            if fld not in s:
                violations.append((idx, sid_val, fld, "missing"))

        if "id" in s and not (isinstance(s["id"], str) and s["id"].strip()):
            violations.append(
                (idx, sid_val, "id",
                 f"must be non-empty str, got {type(s['id']).__name__}: {s['id']!r}")
            )
        if "prompt" in s and not (isinstance(s["prompt"], str) and s["prompt"].strip()):
            violations.append(
                (idx, sid_val, "prompt",
                 f"must be non-empty str, got {type(s['prompt']).__name__}")
            )
        if "expected_vulnerable" in s and not isinstance(s["expected_vulnerable"], bool):
            violations.append(
                (idx, sid_val, "expected_vulnerable",
                 f"must be bool, got {type(s['expected_vulnerable']).__name__}: "
                 f"{s['expected_vulnerable']!r}")
            )

    total = len(samples)
    if violations:
        bad_indices = {idx for idx, _, _, _ in violations}
        valid = total - len(bad_indices)
        pct = 100.0 * valid / total
        preview = "; ".join(
            f"#{idx}(id={sid!r}, field={fld}, reason={rsn})"
            for idx, sid, fld, rsn in violations[:5]
        )
        more = f" ... and {len(violations) - 5} more" if len(violations) > 5 else ""
        raise RuntimeError(
            f"validate_eval_samples: {len(violations)} critical-field violations "
            f"across {total} samples (valid-label rate: {valid}/{total}={pct:.2f}%). "
            f"violations: {preview}{more}"
        )

    pos = sum(1 for s in samples if s["expected_vulnerable"])
    neg = total - pos
    return {
        "total": total,
        "valid": total,
        "pct_valid": 100.0,
        "pos": pos,
        "neg": neg,
    }
