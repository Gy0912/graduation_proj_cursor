"""生成 dataset/ 下 JSONL（仅 SFT/DPO 兼容 demo）。

重要（修复 missing-label 评测 Bug 后）:
  本脚本**不再**写任何评测集到 configs.files.eval_prompts。
  评测集的唯一权威源是 data/combined/eval_fixed.json，**仅**由
  ``scripts/build_eval_fixed.py`` 产出（自 2026-04-20 单一写入者加固起，
  ``dataset/generate_expanded_dataset.py`` / ``write_research_splits`` 都已停止
  写这个文件）；上游只产出 per-task 拆分 ``data/generation/eval.json`` +
  ``data/fix/eval.json``，由 build_eval_fixed.py 合并。
  如果本脚本继续写 configs.files.eval_prompts，将把合成出的无标签 JSONL
  覆盖到 eval_fixed.json，导致回归到原先的破损状态。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from dataset.synthetic_sql import build_synthetic_splits


def save_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    with open(Path(ROOT) / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from training.config_utils import load_merged_config as _lcm
    cfg = _lcm(ROOT, args.config)

    ds = cfg["dataset"]
    files = cfg["files"]
    data = build_synthetic_splits(
        train_n=ds["train_sft_n"],
        val_n=ds["val_sft_n"],
        eval_prompts_n=ds["eval_prompts_n"],
        seed=ds["seed"],
    )

    base = Path(ROOT)
    save_jsonl(base / files["train_sft"], data["train_sft"])
    save_jsonl(base / files["val_sft"], data["val_sft"])
    save_jsonl(base / files["train_dpo"], data["train_dpo"])

    print("[OK] wrote demo SFT/DPO JSONL:")
    for k in ("train_sft", "val_sft", "train_dpo"):
        print(" ", base / files[k])
    print(
        "[skip] eval_prompts intentionally NOT written here; "
        "evaluation loads the labeled dataset at "
        f"{base / files['eval_prompts']} produced by scripts/build_eval_fixed.py."
    )


if __name__ == "__main__":
    main()
