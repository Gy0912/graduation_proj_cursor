"""
(A) LoRA + SFT：在训练循环外完成分词（dataset.map batched），并确保 4bit+GPU 路径正确。
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
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from training.amp_grad_debug import AmpGradDebugCallback
from training.dtype_utils import (
    cast_trainable_bf16_to_float16,
    print_cuda_amp_debug,
    summarize_parameter_dtypes,
)
from training.gpu_debug import GpuDebugCallback
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
        raise RuntimeError(
            "未检测到 CUDA GPU。本脚本禁止 CPU 回退，请使用 NVIDIA 驱动 + CUDA 版 PyTorch。"
        )
    print(f"[device] using GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    # 在构建 TrainingArguments 之前固定 Accelerate 混合精度（见 transformers TrainingArgs 第 5 节）
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    require_cuda()

    base = cfg["model"]["base_model"]
    out_dir = cfg["paths"]["lora_sft_dir"]
    tcfg = cfg["training"]

    # ---------- CUDA / AMP 诊断 ----------
    print_cuda_amp_debug()
    if torch.cuda.is_available():
        print(f"[DEBUG] device count: {torch.cuda.device_count()}")

    # ---------- 数据：instruction / input / output ----------
    files = cfg.get("files", {})
    train_json = files.get("train_sft_json")
    if train_json and (ROOT / train_json).exists():
        data_path = ROOT / train_json
    else:
        data_path = ROOT / files.get("sql_security_dataset", "dataset/sql_security_dataset.json")
    if not data_path.exists():
        raise FileNotFoundError(
            f"未找到 {data_path}。请运行 dataset/generate_expanded_dataset.py 或 dataset/generate_sql_security_dataset.py"
        )
    records = load_sql_security_json(data_path)

    # --- Adversarial pre-flight：禁止脆弱 SQL 流入 SFT target ---
    # 要求训练集中 expected_vulnerable=True 样本的 output 已经被替换为三段式
    # 对抗响应；任何一条含有 SQL 注入模式的 output 都必须在训练启动前被拒绝。
    run_pretraining_sanity_checks(records)

    ds_cfg = cfg.get("dataset", {})
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
    print(f"[data] train={len(train_ds)} val={len(val_ds)} columns={train_ds.column_names}")

    # ---------- 模型：8GB 友好 4bit + fp16 + device_map=auto ----------
    quant = None
    if tcfg.get("load_in_4bit", True):
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
    summarize_parameter_dtypes(model, "base_model_before_lora")

    target_modules = resolve_lora_target_modules(
        model, tcfg.get("lora_target_modules", "auto")
    )
    print(f"[LoRA] target_modules = {target_modules}")

    peft_cfg = LoraConfig(
        r=tcfg["lora_r"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # 先手动挂 LoRA，便于在训练前检查/修正 adapter dtype（避免 bf16 + fp16 GradScaler 冲突）
    model = get_peft_model(model, peft_cfg)
    summarize_parameter_dtypes(model, "after_lora_peft")
    cast_trainable_bf16_to_float16(model)
    summarize_parameter_dtypes(model, "after_bf16_to_fp16_fix")

    # 完全关闭 AMP：fp16/bf16 均为 False，不创建 GradScaler（见 amp_grad_debug 回调验证）
    disable_amp = bool(tcfg.get("disable_amp", True))
    if disable_amp:
        use_fp16 = False
        use_bf16 = False
        print("[amp] disable_amp=True -> fp16=False, bf16=False, ACCELERATE_MIXED_PRECISION=no")
    else:
        use_bf16 = bool(tcfg.get("bf16", False)) and torch.cuda.is_available()
        use_fp16 = bool(tcfg.get("fp16", False)) and torch.cuda.is_available()
        if use_bf16 and use_fp16:
            raise ValueError("不能同时开启 fp16 与 bf16；请只选一种混合精度。")
        if use_bf16 and not torch.cuda.is_bf16_supported():
            print("[WARN] bf16=True 但当前 GPU 不支持 bf16，将改用 fp16。")
            use_bf16 = False
            use_fp16 = True

    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))

    # ---------- SFTConfig ----------
    sft_args = SFTConfig(
        output_dir=str(ROOT / out_dir),
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
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=tcfg.get("dataloader_pin_memory", True) and torch.cuda.is_available(),
        max_length=max_len,
        dataset_kwargs={"skip_prepare_dataset": False},
        report_to=[],
        remove_unused_columns=False,
    )

    amp_dbg = AmpGradDebugCallback(model)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=None,
        callbacks=[GpuDebugCallback(model), amp_dbg],
    )
    amp_dbg.trainer = trainer

    trainer.train()

    Path(ROOT / out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(ROOT / out_dir)
    tok.save_pretrained(ROOT / out_dir)
    print(f"[OK] LoRA SFT saved to {ROOT / out_dir}")


if __name__ == "__main__":
    main()
