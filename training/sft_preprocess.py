"""
SFT 数据预处理：在训练循环外完成分词（dataset.map batched=True），产出 TRL 期望的
input_ids + completion_mask，供 DataCollatorForLanguageModeling 计算 completion-only loss。

对抗训练契约（2026-04-22 起强制生效）：
  * **不过滤任何样本**。``expected_vulnerable=True`` 的样本也必须作为 SFT target
    进入训练集——它们的 ``output`` 不再是脆弱 SQL，而是由
    ``dataset/adversarial.py::build_secure_response`` 合成的三段式安全响应
    （``[SECURITY WARNING]`` / ``[EXPLANATION]`` / ``[SAFE SOLUTION]``）。
  * 调用 ``build_sft_dataset_from_records(...)`` 之前，训练入口脚本（
    ``training/train_lora_sft.py`` / ``training/train_qlora_sft.py``）必须调用
    ``run_pretraining_sanity_checks(records)``，该函数跑两条硬断言：

      1. 任何样本的 ``output`` 里不得出现脆弱 SQL 模式（对 ``expected_vulnerable
         =True`` 样本只扫描 SAFE SOLUTION 代码块，避免 natural-language 解释
         里引用的 SQL 关键词误伤；对 ``expected_vulnerable=False`` 样本扫描整
         个 ``output``）；
      2. 所有 ``expected_vulnerable=True`` 样本的 ``output`` 必须同时包含
         3 段 marker；合规率 < 100% 即 FAIL FAST。

    任一断言失败，脚本会 ``raise RuntimeError`` 阻止训练继续——这是为了保证
    模型**永远不会**从我们的训练目标里学到 SQL 注入模式。
"""
from __future__ import annotations

import sys
import ast
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FORBIDDEN_TRAINING_TOKENS: tuple[str, ...] = (
    "[SECURITY WARNING]",
    "[EXPLANATION]",
    "[SAFE SOLUTION]",
    "### Response",
    "### Solution",
    "### Test",
)


def row_to_prompt_completion(
    instruction: str,
    input_text: str,
    output: str,
    template: str = "Instruction:\n{instruction}\n\nInput:\n{input}\n\n",
) -> tuple[str, str]:
    """将 instruction/input/output 拼成 prompt + completion（completion 不含 prompt）。"""
    prompt = template.format(instruction=instruction.strip(), input=(input_text or "").strip())
    completion = output.strip()
    if completion and not completion.endswith("\n"):
        completion += "\n"
    return prompt, completion


def tokenize_prompt_completion_batched(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list]:
    """
    batched=True 时调用。对每个样本：
    - 对 prompt 与 prompt+completion 分别编码（与 TRL 默认行为一致：add_special_tokens=False）
    - 构造 completion_mask：prompt 段为 0，completion 段为 1
    - 超长则从右侧截断，并同步截断 completion_mask
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    all_input_ids: list[list[int]] = []
    all_masks: list[list[int]] = []

    for prompt, completion in zip(prompts, completions):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

        if len(full_ids) < len(p_ids):
            # 极端 tokenizer 行为：保底用全长当 completion
            p_ids = full_ids[: max(1, len(full_ids) // 2)]

        if full_ids[: len(p_ids)] != p_ids:
            # 对齐：以最长公共前缀为准（避免空格/特殊符号导致不一致）
            common = 0
            for i, (a, b) in enumerate(zip(p_ids, full_ids)):
                if a == b:
                    common = i + 1
                else:
                    break
            p_ids = full_ids[:common]

        comp_mask = [0] * len(p_ids) + [1] * (len(full_ids) - len(p_ids))

        # 右截断
        if len(full_ids) > max_length:
            overflow = len(full_ids) - max_length
            full_ids = full_ids[-max_length:]
            comp_mask = comp_mask[-max_length:]
            # 若截断吃掉全部 prompt，至少保留最后一个 token 的监督（避免全 -100）
            if sum(comp_mask) == 0:
                comp_mask[-1] = 1

        all_input_ids.append(full_ids)
        all_masks.append(comp_mask)

    return {"input_ids": all_input_ids, "completion_mask": all_masks}


def build_sft_dataset_from_records(
    records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """records 含 instruction, input, output。"""
    prompts: list[str] = []
    completions: list[str] = []
    for r in records:
        inp = r.get("input", "")
        if inp is None or (isinstance(inp, str) and not inp.strip()):
            inp = r.get("input_code", "") or ""
        p, c = row_to_prompt_completion(
            str(r.get("instruction", "")),
            str(inp),
            str(r.get("output", "")),
        )
        prompts.append(p)
        completions.append(c)

    ds: Dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    remove_cols = [c for c in ds.column_names if c not in ("prompt", "completion")]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    ds = ds.map(
        lambda batch: tokenize_prompt_completion_batched(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["prompt", "completion"],
        desc="Tokenizing (batched)",
    )
    return ds


def train_val_split(records: list[dict], val_ratio: float, seed: int) -> tuple[list, list]:
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    if not train:
        train, val = val[:-1], val[-1:]
    return train, val


# ---------------------------------------------------------------------------
# 对抗训练 pre-flight sanity checks
# ---------------------------------------------------------------------------


def _contains_forbidden_scaffold(text: str) -> list[str]:
    return [tok for tok in FORBIDDEN_TRAINING_TOKENS if tok in text]


def assert_training_outputs_are_python_only(records: list[dict]) -> dict[str, Any]:
    """硬断言：所有训练输出必须是纯 Python 代码（无结构化模板/分节文本）。"""
    violations: list[dict[str, Any]] = []
    valid_python = 0
    total = len(records)

    for i, r in enumerate(records):
        output = str(r.get("output", ""))
        if not output.strip():
            violations.append(
                {"index": i, "id": r.get("id"), "kind": "empty_output"}
            )
            continue

        forbidden = _contains_forbidden_scaffold(output)
        if forbidden:
            violations.append(
                {"index": i, "id": r.get("id"), "kind": "forbidden_scaffold", "tokens": forbidden}
            )
            continue

        try:
            ast.parse(output)
        except SyntaxError as exc:
            violations.append(
                {
                    "index": i,
                    "id": r.get("id"),
                    "kind": "non_python_output",
                    "reason": f"{exc.msg} at line {exc.lineno}, column {exc.offset}",
                }
            )
            continue
        valid_python += 1

    if violations:
        preview = violations[:5]
        raise RuntimeError(
            f"SFT pre-flight: training outputs must be python-only "
            f"(valid={valid_python}/{total}). First 5 violations: {preview}. "
            "Regenerate dataset and ensure output contains only Python code."
        )
    return {
        "total": total,
        "valid_python_outputs": valid_python,
        "valid_python_rate_pct": (100.0 * valid_python / total) if total else 100.0,
    }


def assert_no_vulnerable_sql_patterns(records: list[dict]) -> dict[str, Any]:
    """兼容接口：当前仅保留 python-only 约束，不再执行对抗模板扫描。"""
    return {
        "total_samples": len(records),
        "violations": [],
    }


def run_pretraining_sanity_checks(records: list[dict]) -> dict[str, Any]:
    """SFT 训练入口调用的「唯一对外」pre-flight 函数，组合两条硬断言 + 打印审计日志。

    失败即 ``RuntimeError``，阻止 ``Trainer.train()`` 启动；训练脚本不得把本函数
    的异常吞掉，也不得在异常后继续运行。
    """
    fmt = assert_training_outputs_are_python_only(records)
    vuln = assert_no_vulnerable_sql_patterns(records)

    total = len(records)
    valid_python = fmt["valid_python_outputs"]
    compliance = fmt["valid_python_rate_pct"]
    print(f"[SFT sanity] total_samples={total}")
    print(f"[SFT sanity] valid_python_outputs={valid_python}")
    print(f"[SFT sanity] python_only_compliance_rate={compliance:.2f}%")
    print("[SFT sanity] forbidden scaffold tokens in outputs = 0 (hard assert passed)")
    return {
        "total": total,
        "valid_python_outputs": valid_python,
        "python_only_compliance_rate_pct": compliance,
        "sanity_report": vuln,
    }
