"""一键顺序执行 7 组实验并生成统一对比。

数据构建阶段明确遵循「``data/combined/eval_fixed.json`` 单一写入者」不变式：

    generate_expanded_dataset.py  -> data/combined/train.json
                                     data/generation/{train,eval}.json
                                     data/fix/{train,eval}.json
    build_dataset.py              -> dataset/*.jsonl (SFT/DPO demo only)
    build_eval_fixed.py           -> data/combined/eval_fixed.json  [唯一写入者]

``scripts/build_eval_fixed.py`` 是 ``eval_fixed.json`` 的**唯一**写入路径，
它读取 per-task 拆分（``data/generation/eval.json`` + ``data/fix/eval.json``）
合并出带完整 ``expected_vulnerable`` / ``vulnerability_type`` / ``difficulty``
标签的评测集。不允许在管线任何其它步骤里创建或覆盖这个文件。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def run_nonblocking(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[WARN] command failed but pipeline continues: {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-lora-dpo", action="store_true")
    p.add_argument("--skip-qlora-dpo", action="store_true")
    args = p.parse_args()

    py = sys.executable
    cfg = args.config

    # [data] 上游：生成 per-task 拆分；此步骤已不再写 eval_fixed.json
    run([py, "dataset/generate_expanded_dataset.py"])
    # [data] SFT/DPO demo JSONL；此步骤不触碰评测集（files.eval_prompts）
    run([py, "scripts/build_dataset.py", "--config", cfg])
    # [data] 唯一写入者：合并 per-task 拆分 -> data/combined/eval_fixed.json
    run([py, "scripts/build_eval_fixed.py"])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "baseline"])

    run([py, "training/train_lora_only.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "lora_only"])
    run([py, "training/train_lora_sft.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "lora_sft"])

    if not args.skip_lora_dpo:
        run([py, "training/dpo_train.py", "--config", "configs/dpo.yaml"])
        run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "lora_dpo"])

    run([py, "training/train_qlora_only.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "qlora_only"])

    run([py, "training/train_qlora_sft.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "qlora_sft"])

    if not args.skip_qlora_dpo:
        run_nonblocking([py, "training/train_qlora_dpo.py", "--config", "configs/dpo.yaml"])
        run_nonblocking(
            [
                py,
                "evaluation/evaluate.py",
                "--config",
                cfg,
                "--model",
                "qlora_dpo",
                "--allow-missing-adapter",
            ]
        )

    run([py, "scripts/compare_results.py", "--config", cfg])


if __name__ == "__main__":
    main()
