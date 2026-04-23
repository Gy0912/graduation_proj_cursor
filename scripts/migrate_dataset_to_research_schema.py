"""
将旧版扁平 JSON（train_expanded / eval_expanded 等）迁移为研究 schema，并写入：

  data/combined/train.json
  data/generation/{train,eval}.json
  data/fix/{train,eval}.json

**本脚本不写 data/combined/eval_fixed.json**。自 2026-04-20 单一写入者加固起，
该文件的写入权已完全移交 ``scripts/build_eval_fixed.py``。本脚本的职责仅限于
把 legacy 扁平格式迁成研究 schema 的 per-task 拆分；评测合并由下游独立步骤完成。

不删除源文件。若缺字段则从 attack_type 等推断；evaluation 样本缺 expected_vulnerable
时会由下游 write_research_splits 的门闸直接抛错（FAIL FAST，不再静默填 False）。

典型用法（PowerShell）::

  # 第一步：迁移 legacy 数据为 per-task 拆分
  .\\.venv\\Scripts\\python.exe scripts\\migrate_dataset_to_research_schema.py \\
      --train data\\train_expanded.json --eval data\\eval_expanded.json

  # 第二步：由 build_eval_fixed.py 产出权威评测集
  .\\.venv\\Scripts\\python.exe scripts\\build_eval_fixed.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.research_schema import to_research_record, write_research_splits


def _normalize_train_row(row: dict) -> dict:
    """
    将 legacy 训练样本规范化为研究 schema 兼容字段。

    **FAIL FAST（2026-04-20 四次加固）**
      - 不再对缺失的 ``expected_vulnerable`` 默认填 False：之前这样做会把未标注的样本
        全部当成「非漏洞」送进 SFT/DPO 训练目标挑选逻辑，污染训练信号。legacy 训练数据
        若缺 label，必须在源头显式标注后再 migrate。
      - 不再对缺失的 ``expected_vulnerable`` 类型做宽容：必须是 Python ``bool``。

    其它元数据（``vulnerability_type``/``task_type``/``difficulty``）仍允许根据
    ``attack_type`` 等派生，因为它们是分组 key，缺失时合理 fallback 不会污染标签。
    """
    if "expected_vulnerable" not in row:
        raise ValueError(
            f"Missing expected_vulnerable in training sample (id={row.get('id')!r})。"
            "FAIL FAST：训练行禁止默认填 False，请在源数据里显式标注；legacy 数据需要先回填标签再 migrate。"
        )
    if not isinstance(row["expected_vulnerable"], bool):
        raise ValueError(
            f"expected_vulnerable 必须是 bool，实际为 "
            f"{type(row['expected_vulnerable']).__name__}: {row['expected_vulnerable']!r} "
            f"(id={row.get('id')!r})"
        )
    out = dict(row)
    if "vulnerability_type" not in out and "attack_type" in out:
        out["vulnerability_type"] = out["attack_type"]
    if "task_type" not in out:
        out["task_type"] = "generation"
    if "difficulty" not in out:
        out["difficulty"] = "medium"
    if "input" in out and "input_code" not in out:
        out["input_code"] = out.get("input")
    return out


def _normalize_eval_row(row: dict) -> dict:
    if "expected_vulnerable" not in row:
        raise ValueError(
            f"Missing expected_vulnerable in evaluation sample (id={row.get('id')!r})。"
            "FAIL FAST: 评测行不允许默认填 False，请在源数据里显式标注。"
        )
    if not isinstance(row["expected_vulnerable"], bool):
        raise ValueError(
            f"expected_vulnerable 必须是 bool，实际为 "
            f"{type(row['expected_vulnerable']).__name__}: {row['expected_vulnerable']!r} "
            f"(id={row.get('id')!r})"
        )
    out = dict(row)
    if "vulnerability_type" not in out and "attack_type" in out:
        out["vulnerability_type"] = out["attack_type"]
    if "task_type" not in out:
        out["task_type"] = "generation"
    if "difficulty" not in out:
        out["difficulty"] = "medium"
    if "input" in out and "input_code" not in out:
        out["input_code"] = out.get("input")
    return out


def load_train_like(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    return [_normalize_train_row(r) for r in data if isinstance(r, dict)]


def load_eval_like(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    out: list[dict] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        er = _normalize_eval_row(r)
        if "instruction" not in er and er.get("prompt"):
            p = str(er["prompt"])
            if "### Instruction:" in p and "### Input:" in p:
                try:
                    rest = p.split("### Instruction:", 1)[1]
                    instr, inp = rest.split("### Input:", 1)
                    inp, _ = inp.split("### Response:", 1)
                    er["instruction"] = instr.strip()
                    er["input"] = inp.strip()
                except ValueError:
                    er.setdefault("instruction", "")
                    er.setdefault("input", "")
            else:
                er.setdefault("instruction", p)
                er.setdefault("input", "")
        out.append(er)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate legacy dataset JSON to research schema paths")
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train_expanded.json")
    p.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_expanded.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将写入的样本数，不写文件",
    )
    args = p.parse_args()

    if not args.train.exists():
        raise FileNotFoundError(args.train)
    if not args.eval.exists():
        raise FileNotFoundError(args.eval)

    train_rows = load_train_like(args.train)
    eval_rows = load_eval_like(args.eval)

    for i, r in enumerate(train_rows):
        if not r.get("instruction") and not r.get("output"):
            print(f"[warn] train row {i} may be incomplete: keys={list(r.keys())}", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] train={len(train_rows)} eval={len(eval_rows)}")
        return

    write_research_splits(train_rows, eval_rows, ROOT)
    print(
        f"[OK] wrote data/combined/train.json, data/generation/*, data/fix/* "
        f"from {args.train} + {args.eval}"
    )
    print(
        "[note] data/combined/eval_fixed.json is NOT produced here; "
        "run `scripts/build_eval_fixed.py` next to merge the per-task eval splits."
    )


if __name__ == "__main__":
    main()
