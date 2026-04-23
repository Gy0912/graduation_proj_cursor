"""训练共享工具：兼容不同 transformers/trl 版本的参数构造。"""
from __future__ import annotations

import inspect
from typing import Any

from transformers import TrainingArguments


def build_training_arguments(**kwargs: Any) -> TrainingArguments:
    """根据当前 TrainingArguments 签名过滤不支持的字段。"""
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)


def sft_format(prompt: str, completion: str) -> str:
    """统一指令格式（论文中写清楚即可复现）。"""
    return f"### Instruction\n{prompt}\n### Response\n{completion}"
