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
    """在标准 DPOTrainer 上增加 logits clamp、loss/梯度 NaN 防护。

    同时覆写 ``_prepare_dataset``，直接用主进程循环调用 TRL 的
    ``tokenize_fn`` 而非 ``dataset.map()``，彻底规避 datasets 多进程
    子进程中 accelerate.logging 的 PartialState 未初始化崩溃。"""

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

    def _prepare_dataset(self, dataset, processing_class, args, mode):
        """主进程同步 tokenization —— 完全替代 datasets.map() 多进程路径。

        TRL 的默认实现在 ``dataset.map()`` 中 spawn 子进程调用 tokenize_fn，
        子进程中 accelerate.logging 因 PartialState 未初始化而崩溃。
        这里逐条在主进程中 tokenize 后重新构建 Dataset，完全回避多进程。
        """
        from datasets import Dataset

        max_len = getattr(args, "max_length", getattr(self, "max_length", 1024))
        max_prompt = getattr(
            args, "max_prompt_length", getattr(self, "max_prompt_length", 512)
        )

        def _tokenize_one(example):
            """与 TRL tokenize_fn 等价的单条 tokenization（主进程运行）。"""
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            prompt_ids = processing_class(prompt, add_special_tokens=False)["input_ids"]
            chosen_ids = processing_class(chosen, add_special_tokens=False)["input_ids"]
            rejected_ids = processing_class(rejected, add_special_tokens=False)["input_ids"]

            # Truncation (与 TRL 逻辑一致)
            if max_prompt is not None and len(prompt_ids) > max_prompt:
                prompt_ids = prompt_ids[-max_prompt:]
            if max_len is not None:
                total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
                if total_len > max_len:
                    # 优先截断 chosen/rejected，保留 prompt
                    budget = max_len - len(prompt_ids)
                    if budget <= 0:
                        prompt_ids = prompt_ids[-max_len:]
                        budget = 0
                    chosen_ids = chosen_ids[:budget]
                    rejected_ids = rejected_ids[:budget]

            # Build prompt + chosen/rejected
            chosen_input_ids = prompt_ids + chosen_ids
            chosen_labels = [-100] * len(prompt_ids) + chosen_ids
            rejected_input_ids = prompt_ids + rejected_ids
            rejected_labels = [-100] * len(prompt_ids) + rejected_ids

            return {
                "chosen_input_ids": chosen_input_ids,
                "chosen_attention_mask": [1] * len(chosen_input_ids),
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids,
                "rejected_attention_mask": [1] * len(rejected_input_ids),
                "rejected_labels": rejected_labels,
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": [1] * len(prompt_ids),
            }

        tokenized_rows = []
        for example in dataset:
            tokenized_rows.append(_tokenize_one(example))

        # 用 from_list 构建 Dataset（避免 from_dict 对不等长 list 的抱怨）
        return Dataset.from_list(tokenized_rows)

    @staticmethod
    @contextlib.contextmanager
    def _safe_tokenize_warning_context():
        """防止 TRL 内部 tokenize_fn 的 logger.warning() 在 datasets.map 多进程 worker
        中触发 accelerate 的 RuntimeError（PartialState 未初始化）。

        策略（多层防御）：
          1) 把 ``trl.dpo_trainer.logger.warning`` 替换为使用标准库 logging 的安全版本；
          2) 强制初始化 ``PartialState``（若尚未初始化），让 accelerate 的 logger check 通过；
          3) 若环境变量 ``DPO_TOKENIZE_SILENT=1``，完全静默 warning。
        """
        # 层 1：确保 PartialState 已初始化（主进程中做一次即可影响后续调用）
        try:
            from accelerate.state import PartialState
            PartialState()
        except Exception:
            pass

        # 层 2：替换 TRL 模块级 logger.warning
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
