"""验证 DPO vs SFT prompt 中的 Input 段内容差异"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.generate_expanded_dataset import training_prompt

# === SFT training prompts ===
train_path = ROOT / "data" / "combined" / "train.json"
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print("=" * 70)
print("SFT 训练数据 prompt 构造（sft_preprocess.build_sft_dataset_from_records）")
print("=" * 70)
# 模拟 build_sft_dataset_from_records 的逻辑
for i in range(3):
    r = train_data[i]
    inp = r.get("input", "")
    if inp is None or (isinstance(inp, str) and not inp.strip()):
        inp = r.get("input_code", "") or ""  # FALLBACK!
    # 用 row_to_prompt_completion 的模板
    prompt = f"Instruction:\n{r['instruction'].strip()}\n\nInput:\n{str(inp).strip()}\n\n"
    input_section = str(inp).strip()
    print(f"\n[SFT sample {i}] Input 段内容 ({len(input_section)} chars):")
    print(f"  {input_section[:150]}...")
    print(f"  Input 段是否为非空: {bool(input_section)}")

# === DPO training prompts ===
dpo_path = ROOT / "data" / "dpo_pairs.json"
with open(dpo_path, "r", encoding="utf-8") as f:
    dpo_lines = [json.loads(line) for line in f if line.strip()]

print("\n" + "=" * 70)
print("DPO 训练数据 prompt（build_dpo_pairs 构造）")
print("=" * 70)
for i in range(3):
    r = dpo_lines[i]
    prompt = r["prompt"]
    # 提取 Input: 后面的内容
    input_marker = "\nInput:\n"
    input_start = prompt.find(input_marker)
    if input_start >= 0:
        input_section = prompt[input_start + len(input_marker):].strip()
    else:
        input_section = ""
    print(f"\n[DPO sample {i}] Input 段内容 ({len(input_section)} chars):")
    if input_section:
        print(f"  {input_section[:150]}...")
    else:
        print(f"  <<EMPTY>>")
    print(f"  Input 段是否为非空: {bool(input_section)}")

# === Evaluation prompts ===
eval_path = ROOT / "data" / "combined" / "eval_fixed.json"
with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print("\n" + "=" * 70)
print("评测数据 prompt（prompt_loader._instruction_input_prompt 构造）")
print("=" * 70)
for i in range(3):
    r = eval_data[i]
    instruction = r.get("instruction", "")
    user_input = r.get("input_code", "") or r.get("input", "")
    prompt = f"Instruction:\n{instruction.strip()}\n\nInput:\n{str(user_input).strip()}\n\n"
    input_section = str(user_input).strip()
    print(f"\n[Eval sample {i}] Input 段内容 ({len(input_section)} chars):")
    print(f"  {input_section[:150]}...")
    print(f"  Input 段是否为非空: {bool(input_section)}")

# === 汇总 ===
print("\n" + "=" * 70)
print("汇总对比")
print("=" * 70)
print(f"""
┌──────────────┬──────────────────────────────────────┐
│ 阶段         │ Input 段内容                         │
├──────────────┼──────────────────────────────────────┤
│ SFT 训练     │ Vulnerable code (完整代码块)          │
│ DPO 训练     │ <<EMPTY>>                            │
│ 评测         │ [EVAL-SET] Schema... attack_hint=... │
└──────────────┴──────────────────────────────────────┘

结论：DPO 训练时的 Input 段为空，但 SFT 训练和评测时 Input 段为非空！
→ DPO 模型从未见过非空 Input 段 → 评测时遇到 [EVAL-SET] 完全困惑 → 退化输出
""")
