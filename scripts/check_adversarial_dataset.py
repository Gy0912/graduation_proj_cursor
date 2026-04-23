r"""对抗训练数据集合规校验 CLI。

本脚本把 ``dataset/adversarial.py::check_adversarial_dataset`` 包装成命令行验证
器，用于在「生成之后、训练之前」独立跑一次硬契约检查：

契约（与训练侧 ``training/sft_preprocess.py::run_pretraining_sanity_checks``
同源）：

  1. 每条 ``expected_vulnerable=True`` 样本的 ``output`` 必须同时含 3 段 marker
     ``[SECURITY WARNING]`` / ``[EXPLANATION]`` / ``[SAFE SOLUTION]``；
  2. SAFE SOLUTION 代码块必须严格参数化：
        - 不含字符串拼接（``"SELECT ..." + x``）；
        - 不含 f-string；
        - 不含 ``.format()`` 或 ``%`` 格式化；
        - 不含 ``sqlalchemy.text(f"...")`` / ``text("...{x}...")`` 这类 ORM 误用；
  3. 每条 ``expected_vulnerable=False`` 样本的 ``output`` 整体也必须通过上述
     脆弱模式扫描（负向回归）。

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

from dataset.adversarial import ADVERSARIAL_MARKERS, check_adversarial_dataset


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
    print(f"[check_adversarial]   adversarial_samples:    {report.adversarial_samples}")
    print(f"[check_adversarial]   format_compliant:       {report.format_compliant}")
    print(
        f"[check_adversarial]   format_compliance_rate: "
        f"{report.format_compliance_rate:.2f}%"
    )
    print(
        f"[check_adversarial]   safe_solution_clean:    "
        f"{report.safe_solution_clean} / {report.adversarial_samples} "
        f"({report.safe_solution_clean_rate:.2f}%)"
    )
    print(
        f"[check_adversarial]   negatives_clean:        "
        f"{report.negative_clean} / {report.negative_samples} "
        f"({report.negative_clean_rate:.2f}%)"
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
    print(f"[check_adversarial]   OK — all contracts satisfied on {path.name}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate adversarial SFT dataset: every expected_vulnerable=True row "
            "must carry the 3-part adversarial markers and a parameterized "
            "SAFE SOLUTION; no row (either class) may contain SQL injection "
            "patterns in its output."
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

    print(f"[check_adversarial] markers enforced: {list(ADVERSARIAL_MARKERS)}")

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
