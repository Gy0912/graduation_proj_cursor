"""DPO 训练数值稳定性：logits 裁剪、NaN 检测。"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import DPOTrainer
from trl.trainer import dpo_trainer as trl_dpo_trainer_module

ROOT = Path(__file__).resolve().parents[1]
NAN_LOG = ROOT / "logs" / "dpo_nan.log"
SAFE_DPO_LOGGER = logging.getLogger("training.stable_dpo_trainer")


def _log_nan_and_exit(reason: str, detail: str = "") -> None:
    NAN_LOG.parent.mkdir(parents=True, exist_ok=True)
    msg = f"[DPO NaN] {reason}"
    if detail:
        msg += f" | {detail}"
    msg += "\n"
    with open(NAN_LOG, "a", encoding="utf-8") as f:
        f.write(msg)
    print(msg, file=sys.stderr)
    raise RuntimeError(msg.strip())


def _clamp_logits_hook(_module, _inputs, output):
    if hasattr(output, "logits") and output.logits is not None:
        output.logits = torch.clamp(output.logits, -20, 20)
        if torch.isnan(output.logits).any():
            _log_nan_and_exit("logits contain NaN after forward")
    return output


def _has_nan_grad(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            return True
    return False


class DpoNanGuardCallback(TrainerCallback):
    """在优化器步进前检查梯度 NaN。"""

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if _has_nan_grad(self._model):
            _log_nan_and_exit("gradients contain NaN")
        return control


class StableDPOTrainer(DPOTrainer):
    """在标准 DPOTrainer 上增加 logits clamp、loss/梯度 NaN 防护。"""

    def __init__(self, *args, **kwargs):
        dpo_args = kwargs.get("args")
        if dpo_args is not None and getattr(dpo_args, "dataset_num_proc", None) not in (None, 1):
            dpo_args.dataset_num_proc = 1
            SAFE_DPO_LOGGER.warning(
                "Overriding dataset_num_proc to 1 for stable DPO preprocessing."
            )
        with self._safe_tokenize_warning_context():
            super().__init__(*args, **kwargs)
        self.model.register_forward_hook(_clamp_logits_hook)
        self.add_callback(DpoNanGuardCallback(self.model))

    @staticmethod
    @contextlib.contextmanager
    def _safe_tokenize_warning_context():
        # DPOTrainer.tokenize_fn warnings run in dataset.map context where accelerate state
        # might be missing (especially on Windows spawn), so avoid accelerate logger calls.
        original_warning = trl_dpo_trainer_module.logger.warning

        def _safe_warning(message, *args, **kwargs):
            if os.getenv("DPO_TOKENIZE_SILENT", "0") == "1":
                return
            SAFE_DPO_LOGGER.warning(message, *args)

        trl_dpo_trainer_module.logger.warning = _safe_warning
        try:
            yield
        finally:
            trl_dpo_trainer_module.logger.warning = original_warning

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        loss = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(loss) and torch.isnan(loss).any():
            _log_nan_and_exit("loss is NaN")
        return out
