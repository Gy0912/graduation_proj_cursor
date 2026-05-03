r"""Code-only 数据集合规校验 CLI。

本脚本把 ``dataset/adversarial.py::check_adversarial_dataset`` 包装成命令行验证器，
用于在「生成之后、训练之前」独立跑一次 code-only 硬契约检查：

契约（与训练侧 ``training/sft_preprocess.py::run_pretraining_sanity_checks``
同源）：

  1. 每条样本的 ``output`` 必须为非空字符串；
  2. ``ast.parse(output)`` 必须通过。

运行（PowerShell）::

    .\.venv\Scripts\python.exe scripts\check_adversarial_dataset.py
    .\.venv\Scripts\python.exe scripts\check_adversarial_dataset.py `
        --input data\train_expanded.json --input data\combined\train.json

退出码：
    0  全部通过
    1  任一样本违反契约
    2  I/O 或参数错误
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.adversarial import check_adversarial_dataset


DEFAULT_INPUTS: tuple[Path, ...] = (
    ROOT / "data" / "train_expanded.json",
    ROOT / "data" / "eval_expanded.json",
    ROOT / "data" / "combined" / "train.json",
)


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: JSON parse error: {e.msg} (line {e.lineno})") from e
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be an array of records")
    return data


def _print_report(path: Path, records: list[dict]) -> int:
    report = check_adversarial_dataset(records)
    print(f"[check_adversarial] file: {path}")
    print(f"[check_adversarial]   total_samples:          {report.total_samples}")
    print(f"[check_adversarial]   parsed_ok:              {report.parsed_ok}")
    print(f"[check_adversarial]   parse_failed:           {report.parse_failed}")
    print(
        f"[check_adversarial]   parse_pass_rate:         "
        f"{report.parse_pass_rate:.2f}%"
    )
    if report.violations:
        print(
            f"[check_adversarial]   violations:             {len(report.violations)}",
            file=sys.stderr,
        )
        for v in report.violations[:20]:
            print(f"[check_adversarial]     - {v}", file=sys.stderr)
        if len(report.violations) > 20:
            print(
                f"[check_adversarial]     ... and {len(report.violations) - 20} more",
                file=sys.stderr,
            )
        return 1
    print(f"[check_adversarial]   OK — all outputs passed ast.parse on {path.name}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate code-only dataset: every row must have non-empty output "
            "and pass ast.parse(output)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        action="append",
        help=(
            "Path to a JSON array dataset to validate. Can be repeated. "
            "Defaults to data/train_expanded.json + data/eval_expanded.json "
            "+ data/combined/train.json."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress OK messages; only print on violation.",
    )
    args = parser.parse_args()

    inputs: list[Path] = list(args.input) if args.input else list(DEFAULT_INPUTS)
    if not inputs:
        print("[check_adversarial] no inputs provided", file=sys.stderr)
        sys.exit(2)

    failures = 0
    for path in inputs:
        try:
            records = _load_json(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"[check_adversarial] skip {path}: {e}", file=sys.stderr)
            failures += 1
            continue
        rc = _print_report(path, records)
        if rc != 0:
            failures += 1

    if failures:
        print(
            f"[check_adversarial] FAIL: {failures} file(s) violated the contract",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[check_adversarial] PASS — every input passed the contract")


if __name__ == "__main__":
    main()
