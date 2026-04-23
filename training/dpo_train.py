"""
在 SFT LoRA 之上做 DPO（TRL DPOTrainer）。需要已训练 outputs/models/lora_sft_starcoder2_3b。

运行：python training/dpo_train.py --config configs/dpo.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig

from training.config_utils import load_merged_config
from training.stable_dpo_trainer import StableDPOTrainer
from training.dtype_utils import cast_trainable_bf16_to_float16, print_cuda_amp_debug, summarize_parameter_dtypes
from training.gpu_debug import GpuDebugCallback


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU。本脚本禁止 CPU 回退，请使用 NVIDIA 驱动 + CUDA 版 PyTorch。"
        )
    print(f"[device] using GPU: {torch.cuda.get_device_name(0)}")


def load_dpo_dataset(path: Path) -> Dataset:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                {
                    "prompt": r["prompt"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                }
            )
    return Dataset.from_list(rows)


def main() -> None:
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()

    cfg = load_merged_config(ROOT, args.config)
    require_cuda()
    print_cuda_amp_debug()

    base = cfg["model"]["base_model"]
    sft_dir = ROOT / cfg["paths"]["lora_sft_dir"]
    out_dir = ROOT / cfg["paths"]["dpo_lora_dir"]
    dpo_path = ROOT / cfg["files"]["dpo_pairs"]
    if not sft_dir.exists():
        raise FileNotFoundError(f"未找到 SFT LoRA：{sft_dir}，请先运行 training/train_lora_sft.py")
    if not dpo_path.exists():
        raise FileNotFoundError(f"未找到 DPO 数据：{dpo_path}，请先运行 dataset/generate_expanded_dataset.py")

    tcfg = cfg["training"]
    dcfg = cfg.get("dpo", {})
    train_ds = load_dpo_dataset(dpo_path)
    print(f"[data] dpo pairs={len(train_ds)}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        dtype=dtype,
        quantization_config=quant,
        device_map="auto",
    )
    print(f"[DEBUG] next(model.parameters()).device = {next(model.parameters()).device}")
    summarize_parameter_dtypes(model, "base_before_peft")

    model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)
    summarize_parameter_dtypes(model, "after_sft_adapter")

    cast_trainable_bf16_to_float16(model)

    max_len = int(dcfg.get("max_length", 768))
    beta = float(dcfg.get("beta", 0.01))

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("grad_accum", 8)),
        learning_rate=float(tcfg["learning_rate_dpo"]),
        num_train_epochs=float(tcfg["num_train_epochs_dpo"]),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 200)),
        save_strategy="steps",
        bf16=False,
        fp16=False,
        max_grad_norm=1.0,
        max_length=max_len,
        beta=beta,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        dataset_num_proc=1,
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = StableDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        processing_class=tok,
        callbacks=[GpuDebugCallback(model)],
    )

    cast_trainable_bf16_to_float16(trainer.model)
    summarize_parameter_dtypes(trainer.model, "dpo_model_after_bf16_fix")

    trainer.train()

    cast_trainable_bf16_to_float16(trainer.model)

    out_dir.mkdir(parents=True, exist_ok=True)
    m = trainer.model
    if isinstance(m, PeftModel) and hasattr(m, "peft_config") and "ref" in getattr(m, "peft_config", {}):
        try:
            m.delete_adapter("ref")
            print("[OK] removed ref adapter before save")
        except Exception as e:
            print(f"[WARN] could not delete ref adapter: {e}")
    m.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[OK] DPO LoRA saved to {out_dir}")


if __name__ == "__main__":
    main()
