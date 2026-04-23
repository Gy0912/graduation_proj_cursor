"""
AMP / GradScaler / 梯度 dtype 调试（在 optimizer.step 之前检查梯度，此时尚未 zero_grad）。
"""
from __future__ import annotations

from collections import Counter

import torch
from transformers import TrainerCallback


class AmpGradDebugCallback(TrainerCallback):
    """
    - on_train_begin: 打印 fp16/bf16 与 Trainer 是否持有 GradScaler（需绑定 trainer）
    - on_pre_optimizer_step: 第一次优化步前打印可训练参数的 grad dtype 分布
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self.trainer = None  # 在 SFTTrainer 创建后赋值: cb.trainer = trainer
        self._logged_grad = False
        self._logged_opt_state = False

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        print(
            f"[amp-debug] TrainingArguments: fp16={getattr(args, 'fp16', None)} "
            f"bf16={getattr(args, 'bf16', None)}"
        )
        print(
            f"[amp-debug] TrainingArguments.mixed_precision={getattr(args, 'mixed_precision', 'n/a')}"
        )
        t = self.trainer
        if t is not None:
            acc = getattr(t, "accelerator", None)
            scaler = getattr(acc, "scaler", None) if acc is not None else None
            native_amp = getattr(acc, "native_amp", None) if acc is not None else None
            print(f"[amp-debug] Accelerator.native_amp={native_amp}")
            print(f"[amp-debug] Accelerator.scaler is None (no GradScaler): {scaler is None}")
            if scaler is not None:
                print(f"[amp-debug] GradScaler._enabled={getattr(scaler, '_enabled', 'n/a')}")
        else:
            print("[amp-debug] (set callback.trainer = trainer 以打印 Accelerate scaler 状态)")

    def on_pre_optimizer_step(self, args, state, control, **kwargs) -> None:
        if self._logged_grad:
            return
        grad_dtypes: Counter[str] = Counter()
        bf16_grad_names: list[str] = []
        n_with_grad = 0
        for name, p in self._model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
            n_with_grad += 1
            dt = str(p.grad.dtype)
            grad_dtypes[dt] += p.grad.numel()
            if p.grad.dtype == torch.bfloat16:
                bf16_grad_names.append(name)
        print("[amp-debug] === first optimizer step: gradient dtype histogram (by numel) ===")
        for k in sorted(grad_dtypes.keys()):
            print(f"    {k}: {grad_dtypes[k]}")
        print(f"[amp-debug] trainable tensors with non-None grad: {n_with_grad}")
        if bf16_grad_names:
            print("[amp-debug] bfloat16 gradients (up to 15 names):")
            for n in bf16_grad_names[:15]:
                print(f"    - {n}")
        else:
            print("[amp-debug] no bfloat16 gradients in sampled trainable params.")
        self._logged_grad = True

    def on_optimizer_step(self, args, state, control, **kwargs) -> None:
        """在第一次 optimizer.step() 之后、global_step 自增之前，state.global_step 仍为 0。"""
        if self._logged_opt_state:
            return
        if state.global_step != 0:
            return
        t = self.trainer
        if t is None or getattr(t, "optimizer", None) is None:
            return
        opt = t.optimizer
        printed = 0
        print("[amp-debug] === optimizer state tensors (first few, after step 0) ===")
        for p in opt.param_groups[0].get("params", []):
            st = opt.state.get(p)
            if not st:
                continue
            for k, v in st.items():
                if torch.is_tensor(v) and printed < 8:
                    print(f"    state['{k}'] dtype={v.dtype} shape={tuple(v.shape)}")
                    printed += 1
            break
        self._logged_opt_state = True
