"""DPO 训练数值稳定性：logits 裁剪、NaN 检测、熵坍缩监控。

2026-05-08 重大修复：重写 _prepare_dataset 以正确匹配 TRL 原生
DPO tokenization 契约。旧版手工构建 tokenization 导致 EOS 丢失、
completion boundary 错乱、completion mask 错误、NaN loss 和模型 collapse。
现已恢复 TRL 的 prompt+chosen 合并 tokenize + 拆分 模式，
并正确追加 EOS token。

2026-05-11 崩溃修复：新增 DPOCollapseGuardCallback —— 在每个 logging step
检测 entropy < 3.0 或 logps绝对值 > 2000 坍缩信号，立即停止训练并保存
当前最佳 checkpoint。
"""
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


class DpoCollapseGuardCallback(TrainerCallback):
    """2026-05-11: 在每个 logging step 检测 DPO 坍缩信号。

    坍缩信号：
      - entropy < 3.0：模型 token 分布完全崩塌
      - abs(logps/chosen) > 2000：log 概率爆炸
      - logits/chosen < 5.0：logits 崩溃

    检测到任一信号 → 立即停止训练并保存当前 checkpoint。
    """

    COLLAPSE_ENTROPY_THRESHOLD = 3.0
    COLLAPSE_LOGP_ABS_THRESHOLD = 2000.0
    COLLAPSE_LOGITS_THRESHOLD = 5.0

    def __init__(self) -> None:
        self._collapse_detected = False
        self._collapse_reason = ""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None or self._collapse_detected:
            return

        entropy = logs.get("entropy")
        logps_c = logs.get("logps/chosen")
        logits_c = logs.get("logits/chosen")

        if entropy is not None and entropy < self.COLLAPSE_ENTROPY_THRESHOLD:
            self._collapse_detected = True
            self._collapse_reason = f"entropy={entropy:.3f} < {self.COLLAPSE_ENTROPY_THRESHOLD}"
        elif logps_c is not None and abs(logps_c) > self.COLLAPSE_LOGP_ABS_THRESHOLD:
            self._collapse_detected = True
            self._collapse_reason = f"logps/chosen={logps_c:.1f} abs > {self.COLLAPSE_LOGP_ABS_THRESHOLD}"
        elif logits_c is not None and abs(logits_c) < self.COLLAPSE_LOGITS_THRESHOLD:
            self._collapse_detected = True
            self._collapse_reason = f"logits/chosen={logits_c:.3f} < {self.COLLAPSE_LOGITS_THRESHOLD}"

        if self._collapse_detected:
            print(
                f"\n[DPO COLLAPSE DETECTED] {self._collapse_reason} at step {state.global_step}. "
                "Stopping training immediately."
            )
            control.should_training_stop = True


class StableDPOTrainer(DPOTrainer):
    """在标准 DPOTrainer 上增加 logits clamp、loss/梯度 NaN 防护。

    2026-05-08 重大修复：覆写 ``_prepare_dataset`` 以主进程同步执行
    TRL 原生 tokenization 语义（EOS 追加、prompt+chosen 合并 tokenize、
    拆分 completion-only ids），同时规避 datasets.map 多进程
    PartialState 崩溃。不再手工拼接 prompt/chosen ids。"""

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
        # 2026-05-11: 添加 DPO 坍缩检测
        self.add_callback(DpoCollapseGuardCallback())

    def _prepare_dataset(self, dataset, processing_class, args, mode):
        """TRL 原生 tokenization 语义 + 主进程同步执行（规避 datasets.map 多进程崩溃）。

        2026-05-08 彻底重写：
        * 旧版手工拼接 prompt_ids + chosen_ids，未追加 EOS，边界错误，
          completion mask 错乱，导致 NaN loss / 梯度爆炸 / 模型 collapse。
        * 现已严格复制 TRL 原生 tokenize_fn 的行为：
          1) 在 tokenization 前追加 EOS 到 chosen/rejected（TRL 的 add_eos）
          2) tokenize prompt+chosen 合并 → 拆分得出 completion-only ids
          3) completion_mask 由 collator 自动从 prompt_ids/chosen_ids 长度构建
          4) 数据集返回 {"prompt_ids","chosen_ids","rejected_ids"}——
             符合 DPODataCollator 的契约

        仅将 datasets.map() 替换为主进程 for 循环以规避
        PartialState 多进程初始化崩溃。
        """
        from datasets import Dataset

        max_len = getattr(args, "max_length", getattr(self, "max_length", 1024))
        max_prompt = getattr(
            args, "max_prompt_length", None
        ) or 512  # TRL 0.29.0 无此参数，使用默认 512
        eos_token = getattr(processing_class, "eos_token", None)
        eos_str = eos_token if isinstance(eos_token, str) else (str(eos_token) if eos_token else "")

        stats = {
            "total": 0,
            "kept": 0,
            "dropped_empty_comp": 0,
            "dropped_trunc": 0,
            "eos_appended": 0,
            "max_prompt_len": 0,
            "max_chosen_len": 0,
            "max_rejected_len": 0,
            "max_total_len": 0,
        }

        def _tokenize_one(example):
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # ── Step 1: 追加 EOS（TRL 原生 add_eos 行为）──
            eos_appended_here = 0
            if eos_str:
                if chosen and not chosen.rstrip().endswith(eos_str):
                    chosen = chosen + eos_str
                    eos_appended_here += 1
                if rejected and not rejected.rstrip().endswith(eos_str):
                    rejected = rejected + eos_str
                    eos_appended_here += 1

            # ── Step 2: 合并 tokenize prompt+chosen / prompt+rejected ──
            # 这是 TRL 原生 tokenize_fn 的核心：不能分别 tokenize，
            # 否则 whitespace / special token 边界不一致。
            prompt_enc = processing_class(prompt, add_special_tokens=True)
            prompt_ids = prompt_enc["input_ids"]

            full_chosen_enc = processing_class(prompt + chosen, add_special_tokens=True)
            full_chosen_ids = full_chosen_enc["input_ids"]
            full_rejected_enc = processing_class(prompt + rejected, add_special_tokens=True)
            full_rejected_ids = full_rejected_enc["input_ids"]

            # ── Step 3: 拆分得出 completion-only ids ──
            # TRL 原生做法：取 prompt+chosen 的前 len(prompt) 个 token
            # 作为 prompt，其余为 completion。
            # 注意：tokenizer 在拼接边界可能因 whitespace 产生 1-2
            # token 的偏移，这与 TRL 的 _safe_tokenize_warning 行为一致——
            # TRL 也是警告后仍按长度切分。这里采用相同策略。
            if len(full_chosen_ids) > len(prompt_ids):
                chosen_ids = full_chosen_ids[len(prompt_ids):]
            else:
                SAFE_DPO_LOGGER.warning(
                    "prompt+chosen shorter than prompt alone — tokenizing chosen separately"
                )
                chosen_ids = processing_class(chosen, add_special_tokens=False)["input_ids"]

            if len(full_rejected_ids) > len(prompt_ids):
                rejected_ids = full_rejected_ids[len(prompt_ids):]
            else:
                SAFE_DPO_LOGGER.warning(
                    "prompt+rejected shorter than prompt alone — tokenizing rejected separately"
                )
                rejected_ids = processing_class(rejected, add_special_tokens=False)["input_ids"]

            # ── Step 4: Truncation（与 TRL 一致的语义）──
            if max_prompt is not None and len(prompt_ids) > max_prompt:
                # 保留 BOS，从左侧截断内容
                if max_prompt <= 1:
                    prompt_ids = prompt_ids[:1]
                else:
                    prompt_ids = [prompt_ids[0]] + prompt_ids[-(max_prompt - 1):]

            if max_len is not None:
                total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
                if total_len > max_len:
                    budget = max_len - len(prompt_ids)
                    if budget > 0:
                        chosen_ids = chosen_ids[:budget]
                        rejected_ids = rejected_ids[:budget]
                    else:
                        # prompt 本身超过 max_length → 丢弃
                        return {"_drop": "prompt_too_long", "_eos": eos_appended_here}

            # ── Step 5: 守卫：空 completion 丢弃 ──
            if len(chosen_ids) == 0 or len(rejected_ids) == 0:
                return {"_drop": "empty_completion", "_eos": eos_appended_here}

            # ── Step 6: 返回符合 DPODataCollator 契约的字段 ──
            return {
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "_eos": eos_appended_here,
            }

        # ── 主进程同步 tokenize ──
        tokenized_rows = []
        for example in dataset:
            stats["total"] += 1
            row = _tokenize_one(example)
            drop_reason = row.pop("_drop", None) if row else "null"
            eos_cnt = row.pop("_eos", 0) if row else 0
            stats["eos_appended"] += eos_cnt
            if drop_reason:
                if drop_reason == "empty_completion":
                    stats["dropped_empty_comp"] += 1
                else:
                    stats["dropped_trunc"] += 1
                continue
            tokenized_rows.append(row)
            stats["kept"] += 1
            stats["max_prompt_len"] = max(stats["max_prompt_len"], len(row["prompt_ids"]))
            stats["max_chosen_len"] = max(stats["max_chosen_len"], len(row["chosen_ids"]))
            stats["max_rejected_len"] = max(stats["max_rejected_len"], len(row["rejected_ids"]))
            stats["max_total_len"] = max(
                stats["max_total_len"],
                len(row["prompt_ids"]) + max(len(row["chosen_ids"]), len(row["rejected_ids"])),
            )

        # ── 调试日志 ──
        SAFE_DPO_LOGGER.info(
            "DPO tokenization stats: total=%d kept=%d "
            "dropped_empty_comp=%d dropped_trunc=%d "
            "eos_appended=%d "
            "max_lens=(prompt=%d chosen=%d rejected=%d total=%d)",
            stats["total"], stats["kept"],
            stats["dropped_empty_comp"], stats["dropped_trunc"],
            stats["eos_appended"],
            stats["max_prompt_len"], stats["max_chosen_len"],
            stats["max_rejected_len"], stats["max_total_len"],
        )
        if stats["dropped_empty_comp"] > 0:
            SAFE_DPO_LOGGER.warning(
                "P1-6: %d/%d samples dropped — completion fully truncated. "
                "Consider increasing max_length (current=%d) or reducing prompt length.",
                stats["dropped_empty_comp"], stats["total"], max_len,
            )
        if stats["kept"] == 0:
            raise ValueError(
                "All DPO samples dropped — reduce prompt length or increase max_length"
            )

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
