"""训练过程 GPU / 显存调试输出。"""
from __future__ import annotations

import torch
from transformers import TrainerCallback


class GpuDebugCallback(TrainerCallback):
    """打印模型所在设备、首步后显存占用（验证是否走 GPU）。"""

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        p = next(self._model.parameters())
        print(f"[DEBUG] first model parameter device: {p.device}")
        if torch.cuda.is_available():
            print(
                f"[DEBUG] torch.cuda.memory_allocated (MB): "
                f"{torch.cuda.memory_allocated() / 1024**2:.2f}"
            )
            print(f"[DEBUG] torch.cuda.get_device_name: {torch.cuda.get_device_name(0)}")

    def on_step_end(self, args, state, control, **kwargs) -> None:
        if state.global_step == 1 and torch.cuda.is_available():
            print(
                f"[DEBUG] after step 1, cuda.memory_allocated (MB): "
                f"{torch.cuda.memory_allocated() / 1024**2:.2f}"
            )
