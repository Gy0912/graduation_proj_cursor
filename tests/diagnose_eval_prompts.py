"""检查训练/评测 prompt 中特殊标记的差异"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.generate_expanded_dataset import training_prompt

# 评测数据
eval_path = ROOT / "data" / "combined" / "eval_fixed.json"
with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"评测样本数: {len(eval_data)}")
print(f"字段: {list(eval_data[0].keys())}")
print()

# 看几个评测 prompt
for i in range(min(3, len(eval_data))):
    r = eval_data[i]
    print(f"=== Eval sample {i} ===")
    print(f"id: {r.get('id')}")
    prompt = r["prompt"]
    print(f"prompt ({len(prompt)} chars):")
    print(prompt[:300])
    print("...")
    print(f"expected_vulnerable: {r.get('expected_vulnerable')}")
    print(f"attack_type: {r.get('attack_type')}")
    print()

# 检查特殊标记
eval_set = sum(1 for r in eval_data if "[EVAL-SET]" in r["prompt"])
eval_unseen = sum(1 for r in eval_data if "[EVAL-UNSEEN]" in r["prompt"])
print(f"评测数据含 [EVAL-SET]: {eval_set}/{len(eval_data)}")
print(f"评测数据含 [EVAL-UNSEEN]: {eval_unseen}/{len(eval_data)}")

# 训练数据
train_path = ROOT / "data" / "combined" / "train.json"
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_set = 0
train_unseen = 0
for r in train_data:
    p = training_prompt(r["instruction"], r.get("input", ""))
    if "[EVAL-SET]" in p:
        train_set += 1
    if "[EVAL-UNSEEN]" in p:
        train_unseen += 1

print(f"训练数据含 [EVAL-SET]: {train_set}/{len(train_data)}")
print(f"训练数据含 [EVAL-UNSEEN]: {train_unseen}/{len(train_data)}")
print()

# 检查评测 prompt 是否以 [EVAL 开头
eval_starts = set()
for r in eval_data:
    inp = r.get("input", "")
    if inp:
        # 取前20字符
        eval_starts.add(inp[:30])
print(f"评测 input 前缀 (去重): {sorted(eval_starts)[:10]}")
