from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.lora_utils import resolve_lora_target_modules


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU。本脚本禁止 CPU 回退。")
    print(f"[device] using GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA only: 仅挂载适配器，不训练。")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    require_cuda()
    tcfg = cfg["training"]
    base = cfg["model"]["base_model"]
    out_dir = ROOT / cfg["paths"]["lora_only_dir"]

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
    target_modules = resolve_lora_target_modules(
        model, tcfg.get("lora_target_modules", "auto")
    )
    peft_cfg = LoraConfig(
        r=tcfg["lora_r"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] LoRA-only adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
