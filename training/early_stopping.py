"""
早停（Early Stopping）与过拟合检测回调（2026-05-10 第十次加固）。

提供：
  - ``EarlyStoppingCallback``：Transformers TrainerCallback，监控 val_loss，
    实现 patience / min_delta 早停 + overfit_ratio 过拟合检测 + 最佳 checkpoint 持久化。
  - ``resolve_best_sft_checkpoint``：供 DPO 训练入口从 SFT 输出目录中定位
    val_loss 最低的 checkpoint（而非最终过拟合 checkpoint）。

设计原则：
  * 与 HuggingFace Trainer / TRL SFTTrainer / DPOTrainer 完全兼容
  * 不修改训练循环代码（纯回调模式）
  * 最佳 checkpoint 通过 ``best_checkpoint.json`` marker 文件记录
  * DPO 侧通过读取 marker 自动选择最佳 checkpoint；无 marker 时回退到最终 checkpoint
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


# ═══════════════════════════════════════════════════════════════
# 早停回调
# ═══════════════════════════════════════════════════════════════

class EarlyStoppingCallback(TrainerCallback):
    """监控 val_loss 的早停回调。

    行为：
      1. 每个 eval 步记录 val_loss 和 train_loss
      2. 若 val_loss 连续 ``patience`` 步未改善（min_delta 容差内）→ 触发早停
      3. 计算 overfit_ratio = train_loss / val_loss，若 < 0.5 且持续恶化 → 打印警告
      4. 每次 val_loss 创历史新低时，保存 checkpoint 到 ``<output_dir>/best_checkpoint/``
         并写入 ``best_checkpoint.json`` marker 文件

    参数：
      patience: 连续未改善步数阈值（默认 5）
      min_delta: 改善的最小绝对变化量（默认 1e-4）
      overfit_warn_threshold: overfit_ratio 低于此值触发警告（默认 0.5）
      overfit_warn_patience: overfit_ratio 连续恶化步数才警告（默认 3）
      save_best: 是否保存最佳 checkpoint 到独立目录（默认 True）
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        overfit_warn_threshold: float = 0.5,
        overfit_warn_patience: int = 3,
        save_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.overfit_warn_threshold = overfit_warn_threshold
        self.overfit_warn_patience = overfit_warn_patience
        self.save_best = save_best

        # 内部状态
        self.best_val_loss: float = float("inf")
        self.best_step: int = -1
        self.steps_without_improvement: int = 0
        self.overfit_worsening_steps: int = 0
        self.overfit_warned: bool = False
        self.early_stopped: bool = False
        self.stopped_at_step: int = -1

        # 日志记录
        self.history: list[dict[str, Any]] = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """每个 eval 步调用。"""
        metrics = kwargs.get("metrics", {})
        if not metrics:
            return

        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        train_loss = None
        # 从最近的日志中提取 train loss
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry and "eval_loss" not in entry:
                    train_loss = entry["loss"]
                    break

        step = state.global_step

        # ── 记录历史 ──
        record = {
            "step": step,
            "epoch": state.epoch,
            "val_loss": val_loss,
            "train_loss": train_loss,
        }
        self.history.append(record)

        # ── 计算 overfit_ratio ──
        overfit_ratio = None
        overfit_status = "normal"
        if train_loss is not None and val_loss > 0:
            overfit_ratio = train_loss / val_loss
            record["overfit_ratio"] = overfit_ratio
            if overfit_ratio < self.overfit_warn_threshold:
                self.overfit_worsening_steps += 1
                overfit_status = "severe" if overfit_ratio < 0.3 else "moderate"
            else:
                self.overfit_worsening_steps = 0
                overfit_status = "normal"

        # ── 过拟合警告 ──
        if (
            self.overfit_worsening_steps >= self.overfit_warn_patience
            and not self.overfit_warned
            and overfit_ratio is not None
        ):
            print(
                f"\n[EARLY STOP] !! OVERFIT WARNING at step {step}: "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"overfit_ratio={overfit_ratio:.4f} (threshold={self.overfit_warn_threshold}) — "
                f"连续 {self.overfit_worsening_steps} 步恶化"
            )
            self.overfit_warned = True

        # ── 最佳 checkpoint 保存 ──
        improved = val_loss < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.best_step = step
            self.steps_without_improvement = 0
            self._save_best_checkpoint(args, state, val_loss, step)
            print(
                f"\n[EARLY STOP] + new best val_loss={val_loss:.6f} at step {step} "
                f"(Δ={self.best_val_loss - val_loss:.2e})"
            )
        else:
            self.steps_without_improvement += 1

        # ── 早停检测 ──
        if self.steps_without_improvement >= self.patience:
            print(
                f"\n[EARLY STOP] STOPPING at step {step} -- "
                f"val_loss 连续 {self.patience} 步未改善 "
                f"(best={self.best_val_loss:.6f} at step {self.best_step}, "
                f"current={val_loss:.6f})"
            )
            self.early_stopped = True
            self.stopped_at_step = step
            control.should_training_stop = True

        # ── 终端日志 ──
        of_str = f" overfit={overfit_ratio:.4f}({overfit_status})" if overfit_ratio is not None else ""
        imp_str = f" patience={self.steps_without_improvement}/{self.patience}" if not improved else ""
        print(
            f"[EARLY STOP] step={step} val_loss={val_loss:.6f} "
            f"best={self.best_val_loss:.6f}@{self.best_step}{of_str}{imp_str}"
        )

    def _save_best_checkpoint(
        self,
        args: TrainingArguments,
        state: TrainerState,
        val_loss: float,
        step: int,
    ):
        """保存最佳 checkpoint 到独立目录并写入 marker 文件。"""
        if not self.save_best:
            return

        best_dir = Path(args.output_dir) / "best_checkpoint"
        best_dir.mkdir(parents=True, exist_ok=True)

        # 写入 marker 文件（供 DPO 侧读取）
        marker = {
            "best_step": step,
            "best_val_loss": val_loss,
            "best_epoch": state.epoch,
            "output_dir": str(best_dir.resolve()),
            "sft_output_dir": args.output_dir,
        }
        marker_path = Path(args.output_dir) / "best_checkpoint.json"
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2)

        # 注意：实际的模型权重由 trainer 自身的 save_steps 机制保存。
        # 这里只记录元数据，DPO 侧通过读取 best_checkpoint.json 知道从
        # 哪个 checkpoint 子目录加载。
        # 但由于 save_steps 可能与 best_step 不对齐，我们在 on_save 中
        # 做最佳权重的显式复制。

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """每次 trainer 保存 checkpoint 时，检查是否匹配最佳步数。"""
        if not self.save_best or self.best_step < 0:
            return

        step = state.global_step
        # 如果当前步数接近最佳步数（在 eval_steps 窗口内），复制到 best_checkpoint/
        if abs(step - self.best_step) <= args.eval_steps:
            best_dir = Path(args.output_dir) / "best_checkpoint"
            # 找到最近保存的 checkpoint 目录
            ckpt_dirs = sorted(
                [d for d in Path(args.output_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if ckpt_dirs:
                latest_ckpt = ckpt_dirs[-1]
                # 复制 adapter 权重文件
                for f in latest_ckpt.iterdir():
                    if f.is_file():
                        dest = best_dir / f.name
                        if not dest.exists() or f.stat().st_mtime > dest.stat().st_mtime:
                            shutil.copy2(str(f), str(dest))
                print(f"[EARLY STOP] copied best checkpoint weights to {best_dir}")


# ═══════════════════════════════════════════════════════════════
# DPO 侧：解析最佳 SFT checkpoint
# ═══════════════════════════════════════════════════════════════

def resolve_best_sft_checkpoint(sft_output_dir: Path) -> tuple[Path, dict | None]:
    """从 SFT 输出目录解析最佳 checkpoint。

    优先级：
      1. 若存在 ``best_checkpoint.json`` marker 且 ``best_checkpoint/`` 目录非空
         → 返回最佳 checkpoint 目录
      2. 否则回退到 sft_output_dir（最终 checkpoint）

    Returns:
      (checkpoint_dir, marker_dict_or_None)
    """
    marker_path = sft_output_dir / "best_checkpoint.json"
    if not marker_path.exists():
        print(f"[EARLY STOP] no best_checkpoint.json found in {sft_output_dir} — "
              f"using final checkpoint (early stopping was not triggered or not configured)")
        return sft_output_dir, None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[EARLY STOP] failed to read best_checkpoint.json: {e} — using final checkpoint")
        return sft_output_dir, None

    best_dir = Path(marker.get("output_dir", ""))
    if not best_dir.exists() or not any(best_dir.iterdir()):
        print(f"[EARLY STOP] best_checkpoint dir {best_dir} is missing or empty — "
              f"using final checkpoint")
        return sft_output_dir, None

    print(
        f"[EARLY STOP] loading best SFT checkpoint from {best_dir} "
        f"(val_loss={marker['best_val_loss']:.6f} at step {marker['best_step']}, "
        f"vs final checkpoint at {sft_output_dir})"
    )
    return best_dir, marker


def print_early_stop_summary(callback: EarlyStoppingCallback) -> None:
    """打印早停摘要。"""
    if not callback.history:
        print("[EARLY STOP] no evaluation history recorded")
        return

    print("\n[EARLY STOP] ======== Summary ========")
    if callback.early_stopped:
        print(f"  Status: EARLY STOPPED at step {callback.stopped_at_step}")
    else:
        print(f"  Status: completed (no early stop triggered)")
    print(f"  Best val_loss: {callback.best_val_loss:.6f} at step {callback.best_step}")
    print(f"  Final val_loss: {callback.history[-1]['val_loss']:.6f} at step {callback.history[-1]['step']}")
    delta = callback.history[-1]['val_loss'] - callback.best_val_loss
    print(f"  val_loss degradation: {delta:+.6f}")
    if callback.overfit_warned:
        print(f"  !! Overfit warning was triggered")

    # 打印最近几轮历史
    print("  Recent history:")
    for r in callback.history[-6:]:
        of = f" overfit_ratio={r.get('overfit_ratio', 'N/A'):.4f}" if r.get('overfit_ratio') is not None else ""
        print(f"    step={r['step']:>5}  val_loss={r['val_loss']:.6f}{of}")
    print("[EARLY STOP] ==========================\n")
