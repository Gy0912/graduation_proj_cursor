"""
训练前 dtype 诊断与一致性修正（避免 FP16 AMP + BF16 张量混用导致
NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda for BFloat16）。
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn


def print_cuda_amp_debug() -> None:
    print(f"[dtype-debug] torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[dtype-debug] torch.version.cuda: {torch.version.cuda}")
        print(
            f"[dtype-debug] torch.cuda.is_bf16_supported(): "
            f"{torch.cuda.is_bf16_supported()}"
        )


def summarize_parameter_dtypes(model: nn.Module, tag: str = "model") -> dict[str, int]:
    """统计 named_parameters 的 dtype 分布。"""
    counts: Counter[str] = Counter()
    bf16_names: list[str] = []
    for name, p in model.named_parameters():
        key = str(p.dtype)
        counts[key] += p.numel()
        if p.dtype == torch.bfloat16:
            bf16_names.append(name)
    print(f"[dtype-debug] === {tag}: parameter dtype histogram (by numel) ===")
    for k in sorted(counts.keys()):
        print(f"    {k}: {counts[k]} elements")
    if bf16_names:
        print(f"[dtype-debug] {tag}: bfloat16 parameter tensors (up to 20 names):")
        for n in bf16_names[:20]:
            print(f"    - {n}")
        if len(bf16_names) > 20:
            print(f"    ... and {len(bf16_names) - 20} more")
    else:
        print(f"[dtype-debug] {tag}: no bfloat16 parameters.")
    return dict(counts)


def cast_trainable_bf16_to_float16(model: nn.Module) -> int:
    """
    将「需要梯度且为 bfloat16」的参数转为 float16。
    在 4bit+LoRA 场景下，部分环境/PEFT 可能把 adapter 建成 bf16，与 fp16 AMP 的 GradScaler 冲突。
    返回被修改的张量数量。
    """
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float16)
            n += 1
    if n:
        print(f"[dtype-fix] casted {n} trainable bfloat16 tensors -> float16")
    return n


def cast_trainable_bf16_to_float32(model: nn.Module) -> int:
    """备选：将可训练 bf16 转为 float32（更稳但更占显存）。"""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float32)
            n += 1
    if n:
        print(f"[dtype-fix] casted {n} trainable bfloat16 tensors -> float32")
    return n
