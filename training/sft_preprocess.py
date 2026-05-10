"""
SFT 数据预处理：在训练循环外完成分词（dataset.map batched=True），产出 TRL 期望的
input_ids + completion_mask，供 DataCollatorForLanguageModeling 计算 completion-only loss。

对抗数据在 **JSON 磁盘形态** 上仍可为三段式
（``[SECURITY WARNING]`` / ``[EXPLANATION]`` / ``[SAFE SOLUTION]`` + fenced Python），
但在进入 ``Trainer`` 之前，``run_pretraining_sanity_checks`` 会先做 **code-only 规范化**：

  * 从每条 ``output`` 中仅抽取 ``[SAFE SOLUTION]`` 内的 Python（
    ``dataset/adversarial.py::extract_code_only_completion``），去掉 markdown fence、
    去掉 warning/explanation 段；对明显「整段重复两遍」的 completion 做折叠；
  * 对抽取结果 ``ast.parse``，无法解析的样本 **丢弃** 并追加写入
    ``logs/sft_code_only_dropout.log``；
  * 规范化后再执行 ``assert_training_outputs_are_python_only``（completion 中不得再
    含三段式 marker 等脚手架）。

这样 SFT 的监督目标与评测侧 **从 raw 输出中抽取再跑 Bandit/规则** 的对象一致（都是
可执行 Python 语义），同时评测仍可单独对完整 ``raw_output`` 做三段式响应质量统计
（评测代码未改，契约在数据与训练侧对齐）。
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from dataset.adversarial import extract_code_only_completion

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
    "# ref=",  # 2026-05-08: 训练数据泄露标志——模型不应输出 ref 注释
)

def normalize_sft_records_for_training(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    原地将每条记录的 ``output`` 替换为 code-only completion；无法抽取或 ``ast.parse``
    失败的样本从列表中移除并记录日志。
    """
    log_path = _ROOT / "logs" / "sft_code_only_dropout.log"
    dropped: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []

    for idx, r in enumerate(records):
        raw = str(r.get("output", ""))
        code = extract_code_only_completion(raw)
        if not code:
            dropped.append({"index": idx, "id": r.get("id"), "reason": "extract_failed_or_empty"})
            continue
        try:
            ast.parse(code)
        except SyntaxError as exc:
            dropped.append(
                {
                    "index": idx,
                    "id": r.get("id"),
                    "reason": f"syntax_error:{exc.msg}:line{exc.lineno}",
                }
            )
            continue
        nr = dict(r)
        nr["output"] = code.rstrip() + "\n"
        kept.append(nr)

    if dropped:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n### UTC {stamp} dropped={len(dropped)} kept={len(kept)}\n")
            for row in dropped[:2000]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    records.clear()
    records.extend(kept)

    return {
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_log_path": str(log_path) if dropped else None,
    }


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
        # P0 FIX (2026-05-08): prompt 需要 BOS token (add_special_tokens=True)
        # 与 DPO P0-3 修复一致。chosen/completion 保持 False 避免中段 BOS。
        p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]

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
    """SFT 训练入口调用的「唯一对外」pre-flight 函数：先 code-only 规范化，再硬断言。

    失败即 ``RuntimeError``，阻止 ``Trainer.train()`` 启动；训练脚本不得把本函数
    的异常吞掉，也不得在异常后继续运行。
    """
    norm = normalize_sft_records_for_training(records)
    print(
        f"[SFT normalize] kept={norm['kept']} dropped={norm['dropped']} "
        f"(code-only SAFE SOLUTION / ast.parse)"
    )
    if norm["drop_log_path"]:
        print(f"[SFT normalize] dropout log: {norm['drop_log_path']}")
    if not records:
        raise RuntimeError(
            "SFT pre-flight: 全部样本在 code-only 规范化中被丢弃（无法抽取 SAFE SOLUTION "
            "或 ast.parse 失败）。请检查数据集与 logs/sft_code_only_dropout.log。"
        )

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
        "normalize_kept": norm["kept"],
        "normalize_dropped": norm["dropped"],
        "normalize_drop_log": norm["drop_log_path"],
    }
