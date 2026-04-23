r"""
构建「带标签」的权威评测集 data/combined/eval_fixed.json。

**本脚本是 ``data/combined/eval_fixed.json`` 的唯一写入者**（自 2026-04-20 单一写入者
加固起）。其它任何数据构建/迁移脚本（如 ``dataset/generate_expanded_dataset.py``、
``scripts/migrate_dataset_to_research_schema.py``、``dataset/research_schema.py::
write_research_splits``）都**不再**写这个文件，避免多写入点导致的 schema 漂移 /
去重规则分叉 / 部分缺标签风险。

背景（修复前的 Bug）:
  旧版 data/combined/eval.json 由 scripts/build_dataset.py 经 synthetic_sql 生成，
  其 schema 仅为 {id, prompt, meta}，不含 expected_vulnerable / vulnerability_type /
  difficulty / task_type。评测侧 prompt_loader._normalize_sample 又用
  `bool(row.get("expected_vulnerable", False))` 静默默认 False，导致：
    - 所有样本均被视为 non-vulnerable；
    - TP = FN = 0，Recall / F1 数学上无效；
    - 评测完全失效，但不会抛错。

本脚本读取数据源（含完整标签）:
  - data/generation/eval.json
  - data/fix/eval.json
进行以下强约束:
  1) 每条样本必须显式含 "expected_vulnerable" 键，且值为 Python bool。
  2) 必须含 "vulnerability_type" / "difficulty" / "task_type" / "instruction" / "id"。
  3) 按 "id" 去重（若 id 重复则保留首次出现；当 id 缺失时按 (instruction, input_code)
     组合键去重）。
  4) 合并后，正类（expected_vulnerable=True）与负类样本数均须 > 0；否则直接 RuntimeError。
  5) **写入前** 对 in-memory 的 ``merged`` 跑 ``_assert_dataset_final``；
     **写入后** 从磁盘读回再次跑 ``_assert_dataset_final``（双重保险，覆盖 JSON
     序列化异常、编码异常、外部手工编辑等场景）。
  6) 以 JSON 数组写入 data/combined/eval_fixed.json（UTF-8, 缩进 2）。

FAIL FAST: 所有校验失败均会抛出异常，不向后兼容旧的「缺标签静默通过」行为。

运行（PowerShell）:
  .\.venv\Scripts\python.exe scripts/build_eval_fixed.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "task_type",
    "instruction",
    "expected_vulnerable",
    "vulnerability_type",
    "difficulty",
)


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"评测源文件不存在: {path}")
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} JSON 解析失败: {e.msg} (line {e.lineno})") from e
    if not isinstance(data, list):
        raise ValueError(f"{path} 必须是 JSON 数组（顶层为列表）")
    return data


def _validate_row(row: dict[str, Any], *, src: str, idx: int) -> None:
    """对每条样本做强 schema 校验；任何缺失或类型错误立即抛错。"""
    if not isinstance(row, dict):
        raise ValueError(
            f"{src} 第 {idx} 条不是 JSON 对象（got {type(row).__name__}）"
        )

    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(
            f"{src} 第 {idx} 条缺失必填字段 {missing}; row id={row.get('id')!r}"
        )

    ev = row["expected_vulnerable"]
    if not isinstance(ev, bool):
        raise ValueError(
            f"{src} 第 {idx} 条 expected_vulnerable 必须是 bool，"
            f"实际为 {type(ev).__name__}: {ev!r}"
        )

    for k in ("task_type", "instruction", "vulnerability_type", "difficulty"):
        v = row[k]
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"{src} 第 {idx} 条 {k} 必须是非空字符串，实际为 {v!r}"
            )

    sid = row["id"]
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError(
            f"{src} 第 {idx} 条 id 必须是非空字符串（不透明哈希，如 'sqlsec-<hex>'），"
            f"实际为 {type(sid).__name__}: {sid!r}"
        )


def _dedup_key(row: dict[str, Any]) -> tuple:
    sid = row.get("id")
    if isinstance(sid, str) and sid.strip():
        return ("id", sid)
    instr = str(row.get("instruction", "")).strip()
    inp = str(row.get("input_code") or row.get("input") or "").strip()
    return ("content", instr, inp)


def merge_eval_sources(
    sources: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: dict[tuple, str] = {}
    per_source_kept: dict[str, int] = {}
    per_source_dups: dict[str, int] = {}

    for src_path in sources:
        rows = _load_json_array(src_path)
        kept = 0
        dup = 0
        for idx, row in enumerate(rows):
            _validate_row(row, src=str(src_path), idx=idx)
            key = _dedup_key(row)
            if key in seen:
                dup += 1
                continue
            seen[key] = str(src_path)
            merged.append(row)
            kept += 1
        per_source_kept[str(src_path)] = kept
        per_source_dups[str(src_path)] = dup

    n_pos = sum(1 for r in merged if r["expected_vulnerable"])
    n_neg = len(merged) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"合并后的评测集只存在一类样本 (pos={n_pos}, neg={n_neg})。"
            "评测指标（Recall/F1/FPR/FNR）数学上无效，FAIL FAST。"
        )

    stats = {
        "total": len(merged),
        "positive": n_pos,
        "negative": n_neg,
        "per_source_kept": per_source_kept,
        "per_source_dups": per_source_dups,
    }
    return merged, stats


SCHEMA_CONSISTENCY_KEYS: tuple[str, ...] = (
    "expected_vulnerable",
    "vulnerability_type",
    "difficulty",
)


def _assert_dataset_final(dataset: list[dict[str, Any]], *, stage: str) -> None:
    """
    **数据集产生之后** 的强校验门闸。承接 [GOAL]：
      - 所有评测样本必须含 ``expected_vulnerable`` 标签；
      - schema 必须齐整：``expected_vulnerable`` / ``vulnerability_type`` /
        ``difficulty`` 三个字段一条不落。

    参数 ``stage`` 用于错误信息区分「写入前 (pre_write)」还是「写入后 (post_write)」，
    便于故障排查时快速定位：pre_write 失败 = merge 阶段就污染了；post_write 失败 =
    磁盘序列化 / 编码异常 / 外部手工编辑。

    任一规则失败立即 ``RuntimeError``；**不**使用 try/except 静默兜底。
    """
    if not dataset:
        raise RuntimeError(
            f"[{stage}] Invalid eval dataset: empty dataset (0 samples)。"
            "build_eval_fixed 期望至少 1 正 + 1 负样本，空集不可能产生有效指标。"
        )

    # [TASK 3] 核心标签存在性
    for idx, sample in enumerate(dataset):
        if "expected_vulnerable" not in sample:
            raise RuntimeError(
                f"[{stage}] Invalid eval dataset: missing label "
                f"(idx={idx}, id={sample.get('id')!r})"
            )

    # [TASK 4] 完整 schema 一致性
    missing_by_idx: list[tuple[int, Any, list[str]]] = []
    for idx, sample in enumerate(dataset):
        missing = [k for k in SCHEMA_CONSISTENCY_KEYS if k not in sample]
        if missing:
            missing_by_idx.append((idx, sample.get("id"), missing))
    if missing_by_idx:
        preview = ", ".join(
            f"idx={i}(id={sid!r}, missing={m})"
            for i, sid, m in missing_by_idx[:5]
        )
        more = (
            f" ... and {len(missing_by_idx) - 5} more"
            if len(missing_by_idx) > 5
            else ""
        )
        raise RuntimeError(
            f"[{stage}] Inconsistent eval schema: {len(missing_by_idx)} samples "
            f"missing one of {list(SCHEMA_CONSISTENCY_KEYS)}. {preview}{more}"
        )

    # 正负双类必须都存在（与 merge 阶段的 pos/neg 检查形成冗余保险）
    n_pos = sum(1 for s in dataset if s["expected_vulnerable"] is True)
    n_neg = sum(1 for s in dataset if s["expected_vulnerable"] is False)
    if n_pos + n_neg != len(dataset):
        raise RuntimeError(
            f"[{stage}] Invalid eval dataset: expected_vulnerable must be strictly "
            f"bool True/False on every sample; got pos={n_pos}, neg={n_neg}, "
            f"total={len(dataset)} (difference = likely int/str leaked in)"
        )
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"[{stage}] Invalid eval dataset: only one class present "
            f"(pos={n_pos}, neg={n_neg})"
        )


def write_eval_fixed(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _readback_and_verify(out_path: Path) -> int:
    """从磁盘读回 eval_fixed.json 并重新跑一次强校验，覆盖 pre-write 检查之外的
    所有风险（JSON 序列化、UTF-8 编码、他人的手工编辑、磁盘写坏）。返回样本数。"""
    if not out_path.exists():
        raise RuntimeError(
            f"eval_fixed.json 未成功写入: {out_path} 不存在（write_eval_fixed 后应存在）。"
        )
    text = out_path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"写入后读回 eval_fixed.json 失败 ({out_path}): {e.msg} "
            f"(line {e.lineno}, column {e.colno})。磁盘序列化异常。"
        ) from e
    if not isinstance(data, list):
        raise RuntimeError(
            f"写入后读回 eval_fixed.json，顶层必须是 JSON 数组，"
            f"实际为 {type(data).__name__}"
        )
    _assert_dataset_final(data, stage="post_write")
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并 data/generation/eval.json + data/fix/eval.json "
        "为 data/combined/eval_fixed.json（带强 schema 校验；单一写入者）"
    )
    parser.add_argument(
        "--generation",
        type=Path,
        default=ROOT / "data" / "generation" / "eval.json",
        help="生成任务评测集路径",
    )
    parser.add_argument(
        "--fix",
        type=Path,
        default=ROOT / "data" / "fix" / "eval.json",
        help="修复任务评测集路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "combined" / "eval_fixed.json",
        help="输出路径，默认 data/combined/eval_fixed.json",
    )
    args = parser.parse_args()

    sources = [args.generation, args.fix]
    merged, stats = merge_eval_sources(sources)

    _assert_dataset_final(merged, stage="pre_write")
    write_eval_fixed(merged, args.out)
    n_readback = _readback_and_verify(args.out)

    print(f"[build_eval_fixed] sources: {[str(p) for p in sources]}")
    print(
        f"[build_eval_fixed] kept per source: {stats['per_source_kept']}; "
        f"dups skipped: {stats['per_source_dups']}"
    )
    print(
        f"[build_eval_fixed] total={stats['total']} "
        f"pos={stats['positive']} neg={stats['negative']}"
    )
    print(f"[build_eval_fixed] wrote {args.out}")
    print(
        f"[build_eval_fixed] post-write readback verified: {n_readback} samples "
        f"(expected_vulnerable / vulnerability_type / difficulty all present)"
    )


if __name__ == "__main__":
    main()
