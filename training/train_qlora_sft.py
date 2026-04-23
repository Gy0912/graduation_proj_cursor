from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dtype_utils import cast_trainable_bf16_to_float16
from training.lora_utils import resolve_lora_target_modules
from training.sft_preprocess import (
    build_sft_dataset_from_records,
    run_pretraining_sanity_checks,
    train_val_split,
)


def load_sql_security_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("训练数据 JSON 应为数组")
    return data


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU。本脚本禁止 CPU 回退。")
    print(f"[device] using GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    parser = argparse.ArgumentParser(description="QLoRA + SFT")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    require_cuda()
    tcfg = cfg["training"]
    ds_cfg = cfg.get("dataset", {})
    files = cfg.get("files", {})
    base = cfg["model"]["base_model"]
    out_dir = ROOT / cfg["paths"]["qlora_sft_dir"]

    train_json = files.get("train_sft_json")
    if train_json and (ROOT / train_json).exists():
        data_path = ROOT / train_json
    else:
        data_path = ROOT / files.get("sql_security_dataset", "dataset/sql_security_dataset.json")
    records = load_sql_security_json(data_path)

    # --- Adversarial pre-flight（与 train_lora_sft.py 对齐）：禁止脆弱 SQL 流入 SFT target ---
    run_pretraining_sanity_checks(records)

    train_records, val_records = train_val_split(
        records,
        val_ratio=ds_cfg.get("val_ratio", 0.1),
        seed=ds_cfg.get("seed", 42),
    )

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_len = tcfg["max_seq_len"]
    train_ds = build_sft_dataset_from_records(train_records, tok, max_len)
    val_ds = build_sft_dataset_from_records(val_records, tok, max_len)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        dtype=torch.float16,
        quantization_config=quant,
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
    cast_trainable_bf16_to_float16(model)

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(tcfg["learning_rate_sft"]),
        num_train_epochs=tcfg["num_train_epochs_sft"],
        warmup_ratio=tcfg["warmup_ratio"],
        logging_steps=tcfg["logging_steps"],
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_strategy="steps",
        eval_strategy="steps",
        bf16=False,
        fp16=False,
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=tcfg.get("dataloader_pin_memory", True) and torch.cuda.is_available(),
        max_length=max_len,
        dataset_kwargs={"skip_prepare_dataset": False},
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=None,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] QLoRA SFT saved to {out_dir}")


if __name__ == "__main__":
    main()
