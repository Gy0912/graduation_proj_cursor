from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import run_eval_always_safe, run_eval_on_prompts, save_results
from evaluation.experiment_log import setup_file_logging
from evaluation.prompt_loader import load_eval_prompts, validate_eval_samples


SUPPORTED_MODELS = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
    "always_safe_model",
)


def resolve_eval_plan(cfg: dict, model_name: str) -> tuple[str | None, bool, str]:
    paths = cfg["paths"]
    outs = cfg["outputs"]
    if model_name == "baseline":
        return None, False, outs["baseline_results"]
    if model_name == "lora_only":
        return paths["lora_only_dir"], False, outs["lora_only_results"]
    if model_name == "lora_sft":
        return paths["lora_sft_dir"], False, outs["lora_sft_results"]
    if model_name == "lora_dpo":
        return paths["dpo_lora_dir"], False, outs["lora_dpo_results"]
    if model_name == "qlora_only":
        return paths["qlora_only_dir"], True, outs["qlora_only_results"]
    if model_name == "qlora_sft":
        return paths["qlora_sft_dir"], True, outs["qlora_sft_results"]
    if model_name == "qlora_dpo":
        return paths["qlora_dpo_dir"], True, outs["qlora_dpo_results"]
    if model_name == "always_safe_model":
        return None, False, outs.get("always_safe_results", "outputs/always_safe_results.json")
    raise ValueError(f"unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评测入口（7种实验设置）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="评测 batch size（覆盖配置 eval.per_device_eval_batch_size）",
    )
    parser.add_argument(
        "--allow-missing-adapter",
        action="store_true",
        help="若适配器不存在则给出警告并退出（码 0）",
    )
    parser.add_argument(
        "--disable-rule-based",
        action="store_true",
        help="关闭规则层，仅依赖 Bandit",
    )
    parser.add_argument(
        "--disable-fallback-detector",
        action="store_true",
        help="已弃用：等同于 --disable-rule-based",
    )
    parser.add_argument(
        "--merge-mode",
        default=None,
        choices=("or", "or_bandit_any", "weighted"),
        help="覆盖配置 eval.merge_mode：合并 Bandit 与规则层的方式",
    )
    parser.add_argument(
        "--enable-taint",
        action="store_true",
        help="启用动态污点追踪（sqlite3 execute 探针，较慢）",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="若设置，将评测运行日志写入该目录（如 logs/experiments）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="覆盖输出 JSON 路径（默认从 config.outputs 读取，消融实验用）",
    )
    args = parser.parse_args()

    # 2026-05-13: 合并 default.yaml（消融实验兼容）
    from training.config_utils import load_merged_config
    cfg = load_merged_config(ROOT, args.config)

    if args.log_dir:
        setup_file_logging(ROOT / args.log_dir, name=f"eval_{args.model}")

    adapter_path, load_in_4bit, output_path = resolve_eval_plan(cfg, args.model)
    # 2026-05-14: 消融实验支持 — --output 覆盖默认路径
    if args.output:
        output_path = args.output
    if args.model != "always_safe_model" and adapter_path is not None and not (ROOT / adapter_path).exists():
        msg = (
            f"[WARN] adapter not found for {args.model}: {ROOT / adapter_path}. "
            "Skip evaluation."
        )
        if args.allow_missing_adapter:
            print(msg)
            return
        raise FileNotFoundError(msg)

    files = cfg["files"]
    gen = cfg["generation"]
    repetition_penalty = float(gen.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(gen.get("no_repeat_ngram_size", 0))
    ev_cfg = cfg.get("eval", {})
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else ev_cfg.get("per_device_eval_batch_size", 4)
    )
    num_workers = int(ev_cfg.get("dataloader_num_workers", 2))
    pin_memory = bool(ev_cfg.get("dataloader_pin_memory", True))
    merge_mode = str(args.merge_mode or ev_cfg.get("merge_mode", "or"))
    enable_taint = bool(args.enable_taint or ev_cfg.get("enable_taint", False))
    code_only_training = bool(ev_cfg.get("code_only_training", True))
    disable_rules = bool(
        args.disable_rule_based or args.disable_fallback_detector
    )
    enable_rule_based = not disable_rules
    eval_samples = load_eval_prompts(ROOT / files["eval_prompts"])

    # [pre-eval validation step] 所有 critical field 必须在每条样本上齐全。
    # 这一步在模型加载之前调用，避免浪费 GPU 初始化时间；evaluator 内部还会再做
    # 一次完整 sanity check（含正负双类），两处 FAIL FAST 形成双重保险。
    stats = validate_eval_samples(eval_samples)
    print(
        f"[Eval] pre-check: total={stats['total']}, valid={stats['valid']} "
        f"({stats['pct_valid']:.2f}%), pos={stats['pos']}, neg={stats['neg']}"
    )

    if args.model == "always_safe_model":
        bundle = run_eval_always_safe(
            samples=eval_samples,
            merge_mode=merge_mode,
            enable_rule_based=enable_rule_based,
            enable_taint=enable_taint,
            code_only_training=code_only_training,
        )
    else:
        bundle = run_eval_on_prompts(
            samples=eval_samples,
            base_model=cfg["model"]["base_model"],
            max_new_tokens=gen["max_new_tokens"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            load_in_4bit=load_in_4bit,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            adapter_path=str(ROOT / adapter_path) if adapter_path else None,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=pin_memory,
            debug_timing=True,
            merge_mode=merge_mode,
            enable_rule_based=enable_rule_based,
            enable_taint=enable_taint,
            code_only_training=code_only_training,
        )
    meta = {
        "mode": args.model,
        "base_model": cfg["model"]["base_model"],
        "adapter_path": str(ROOT / adapter_path) if adapter_path else None,
        "load_in_4bit": load_in_4bit,
        "per_device_eval_batch_size": batch_size,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "config": args.config,
        "eval_dataset": files["eval_prompts"],
        "merge_mode": merge_mode,
        "enable_rule_based": enable_rule_based,
        "enable_taint": enable_taint,
        "always_safe_stub": args.model == "always_safe_model",
    }
    save_results(ROOT / output_path, bundle, meta)
    print(f"[OK] wrote {ROOT / output_path}")


if __name__ == "__main__":
    main()
