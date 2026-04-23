"""
LoRA target_modules 自动推断（按实际模型中的 nn.Linear 子模块后缀匹配）。

StarCoder2（Starcoder2ForCausalLM）在 Transformers 中的实现：
- 注意力：Mistral/Llama 风格 → q_proj, k_proj, v_proj, o_proj
- MLP：类 GPT-2 → c_fc, c_proj

这与 GPT-NeoX（query_key_value、dense_h_to_4h 等）不同。
"""
from __future__ import annotations

import torch.nn as nn

# 常见因果 LM 的 LoRA 目标优先级（从上到下匹配「存在于模型中的」后缀）
_ORDERED_LORA_TARGETS: tuple[str, ...] = (
    # Llama / Mistral / StarCoder2 注意力
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # StarCoder2 / GPT-2 风格 MLP
    "c_fc",
    "c_proj",
    # Llama 风格 MLP
    "gate_proj",
    "up_proj",
    "down_proj",
    # GPT-NeoX 风格（若你换 NeoX 系模型会自动命中）
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
)


def collect_linear_module_suffixes(model: nn.Module) -> set[str]:
    """收集模型中所有 nn.Linear 层的「最后一级名字」。"""
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffixes.add(name.split(".")[-1])
    return suffixes


def get_lora_target_modules(
    model: nn.Module,
    *,
    ordered_candidates: tuple[str, ...] | None = None,
    exclude_suffixes: frozenset[str] | None = None,
) -> list[str]:
    """
    根据模型实际包含的 Linear 层后缀，返回适合 PEFT LoRA 的 target_modules 列表。

    - 优先使用 _ORDERED_LORA_TARGETS 中「在模型里存在」的项，保持可预期顺序。
    - 若一个都匹配不上（极少见），回退为「所有 Linear 后缀去掉 lm_head 等」。
    """
    candidates = ordered_candidates or _ORDERED_LORA_TARGETS
    exclude_suffixes = exclude_suffixes or frozenset({"lm_head"})
    present = collect_linear_module_suffixes(model)

    chosen = [name for name in candidates if name in present and name not in exclude_suffixes]
    if chosen:
        return chosen

    # 回退：所有非排除后缀（仍按字母序，便于论文记录）
    rest = sorted(p for p in present if p not in exclude_suffixes)
    if not rest:
        raise ValueError("未在模型中找到任何 nn.Linear 层，请检查模型加载是否成功。")
    return rest


def resolve_lora_target_modules(
    model: nn.Module,
    config_value: str | list[str] | None,
) -> list[str]:
    """
    解析配置：
    - None / \"auto\"：全自动
    - list：若某项在模型中不存在，打印警告并回退 auto（避免 PEFT 直接报错）
    """
    if config_value in (None, "auto"):
        return get_lora_target_modules(model)

    if not isinstance(config_value, list):
        raise TypeError("lora_target_modules 必须是 list 或 \"auto\"")

    present = collect_linear_module_suffixes(model)
    missing = [m for m in config_value if m not in present]
    if missing:
        auto = get_lora_target_modules(model)
        print(
            f"[LoRA WARN] 配置中的 target_modules 在模型中不存在: {missing}。\n"
            f"            模型中实际存在的 Linear 后缀示例: {sorted(present)[:40]}...\n"
            f"            已改用自动推断: {auto}"
        )
        return auto

    return list(config_value)
