"""
研究用数据集样本 schema 转换与稳定 id。

规范字段：id, task_type, instruction, input_code, expected_vulnerable,
vulnerability_type, difficulty；训练 split 另含 output。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_sample_id(row: dict[str, Any]) -> str:
    """由 instruction + input + task_type + vulnerability_type 生成稳定短 id。"""
    instr = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", row.get("input_code", "")) or "").strip()
    task = str(row.get("task_type", "")).strip()
    vuln_t = str(row.get("attack_type", row.get("vulnerability_type", ""))).strip()
    payload = f"{task}\n{vuln_t}\n{instr}\n{inp}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"sqlsec-{h}"


def to_research_record(
    row: dict[str, Any],
    *,
    include_output: bool = True,
) -> dict[str, Any]:
    """
    将原始样本规范化为研究 schema 记录。

    **FAIL FAST 契约（2026-04-20 四次加固：训练端也同步加严）**
      - 评测记录（``include_output=False``）必须显式包含 bool 类型的 expected_vulnerable。
      - 训练记录（``include_output=True``）同样必须显式包含 bool 类型的 expected_vulnerable。
        之前这里是 ``bool(row.get("expected_vulnerable", False))``——上游如果漏标，
        训练目标会被当作「非漏洞」来挑选 chosen/rejected，SFT target code 与 DPO 偏好
        对都可能错位；因此训练端也改为 FAIL FAST 以杜绝这类静默破坏。

    任一违规均 raise ``ValueError``，错误信息含 id 便于溯源。
    """
    raw_input = row.get("input", row.get("input_code"))
    if raw_input is None:
        inp_code = None
    else:
        s = str(raw_input).strip()
        inp_code = s if s else None

    attack = str(row.get("attack_type", row.get("vulnerability_type", "unknown")))

    if "expected_vulnerable" not in row:
        kind = "training" if include_output else "evaluation"
        raise ValueError(
            f"Missing expected_vulnerable in {kind} sample "
            f"(id={row.get('id')!r}). "
            "训练与评测样本一律禁止默认值回退；请在源数据里显式标注 bool 类型标签。"
        )
    ev_raw = row["expected_vulnerable"]
    if not isinstance(ev_raw, bool):
        raise ValueError(
            f"expected_vulnerable 必须是 bool，实际为 "
            f"{type(ev_raw).__name__}: {ev_raw!r} (id={row.get('id')!r})"
        )
    expected_vulnerable = ev_raw

    raw_id = row.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        rec_id = raw_id
    else:
        rec_id = stable_sample_id(
            {
                **row,
                "attack_type": attack,
                "input": raw_input or "",
            }
        )

    rec: dict[str, Any] = {
        "id": rec_id,
        "task_type": str(row.get("task_type", "generation")),
        "instruction": str(row.get("instruction", "")),
        "input_code": inp_code,
        "expected_vulnerable": expected_vulnerable,
        "vulnerability_type": attack,
        "difficulty": str(row.get("difficulty", "medium")),
    }
    if include_output and "output" in row:
        rec["output"] = str(row.get("output", ""))
    return rec


def split_by_task(rows: Iterable[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    gen: list[dict] = []
    fix: list[dict] = []
    for r in rows:
        if str(r.get("task_type")) == "fix":
            fix.append(r)
        else:
            gen.append(r)
    return gen, fix


def write_research_splits(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    """
    写入研究 schema 数据集：
      - data/combined/train.json  : 全部训练样本（含 output）
      - data/generation/{train,eval}.json / data/fix/{train,eval}.json : 按 task_type 拆分

    **不写 data/combined/eval_fixed.json**。该文件是评测的唯一权威源，必须且仅能
    由 ``scripts/build_eval_fixed.py`` 以 ``data/generation/eval.json`` +
    ``data/fix/eval.json`` 为输入合并产出（单一写入者不变式，由 2026-04-20 加固）。
    本函数不再兼职评测合并文件的写出，否则会出现两条写入路径，让 schema 漂移、
    去重规则分叉的风险重新回来。

    注意（修复 missing-label 评测 Bug 后）:
      1) 不再写 data/combined/eval.json 或 data/combined/eval_fixed.json。
         前者的 JSONL 曾被 build_dataset 覆盖失去标签；后者的写入权已移交
         build_eval_fixed.py。
      2) 写出前对每条评测样本做 expected_vulnerable 存在性 & bool 类型校验；
         任一样本缺标签即 raise ValueError（FAIL FAST）。同时校验正负两类均非空。
      3) per-task 拆分 ``generation/eval.json`` / ``fix/eval.json`` 是
         build_eval_fixed.py 的**上游输入**，因此这里的校验等效于把门闸向数据流
         上游前移一格。
    """
    combined = root / "data" / "combined"
    gen_dir = root / "data" / "generation"
    fix_dir = root / "data" / "fix"
    for d in (combined, gen_dir, fix_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_r = [to_research_record(r, include_output=True) for r in train_rows]
    eval_r = [to_research_record(r, include_output=False) for r in eval_rows]

    _assert_eval_rows_labeled(eval_r)

    with open(combined / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_r, f, ensure_ascii=False, indent=2)

    tg, tf = split_by_task(train_r)
    eg, ef = split_by_task(eval_r)
    with open(gen_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(tg, f, ensure_ascii=False, indent=2)
    with open(gen_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eg, f, ensure_ascii=False, indent=2)
    with open(fix_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(tf, f, ensure_ascii=False, indent=2)
    with open(fix_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(ef, f, ensure_ascii=False, indent=2)


def _assert_eval_rows_labeled(eval_r: list[dict[str, Any]]) -> None:
    """写出前门闸：评测样本（含 per-task 拆分）必须每条都带 bool 类型的
    ``expected_vulnerable`` 且两类均存在；为 build_eval_fixed.py 提供干净上游。"""
    if not eval_r:
        raise ValueError(
            "评测样本列表为空，无法写出 per-task 评测拆分。"
            "build_eval_fixed.py 无法从空输入合并出 eval_fixed.json。"
        )
    for idx, row in enumerate(eval_r):
        if "expected_vulnerable" not in row:
            raise ValueError(
                f"Missing expected_vulnerable in evaluation sample idx={idx} "
                f"(id={row.get('id')!r})"
            )
        if not isinstance(row["expected_vulnerable"], bool):
            raise ValueError(
                f"expected_vulnerable 必须是 bool，idx={idx} 实际为 "
                f"{type(row['expected_vulnerable']).__name__}: {row['expected_vulnerable']!r}"
            )
    n_pos = sum(1 for r in eval_r if r["expected_vulnerable"])
    n_neg = len(eval_r) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Invalid evaluation dataset: only one class present "
            f"(pos={n_pos}, neg={n_neg})"
        )
