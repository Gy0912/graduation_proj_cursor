"""诊断 SFT 训练数据格式"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 检查 SFT 训练数据
train_path = ROOT / "data" / "combined" / "train.json"
if not train_path.exists():
    train_path = ROOT / "data" / "train_expanded.json"

print(f"Reading: {train_path}")
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print(f"Total SFT samples: {len(train_data)}")
print(f"Keys: {list(train_data[0].keys())}")
print()

# 检查 output 格式
code_only = 0
has_fence = 0
has_marker = 0
for i, r in enumerate(train_data[:200]):
    out = r.get("output", "")
    stripped = out.strip()
    if stripped.startswith("import ") or stripped.startswith("from "):
        code_only += 1
    if "```python" in out or "```" in out:
        has_fence += 1
    if "[SAFE SOLUTION]" in out or "[EXPLANATION]" in out:
        has_marker += 1

print(f"SFT output 格式 (前200条):")
print(f"  纯代码开头 (import/from): {code_only}/200")
print(f"  含 markdown fence: {has_fence}/200")
print(f"  含 SAFE SOLUTION marker: {has_marker}/200")
print()

# 看样例
for idx in range(min(3, len(train_data))):
    r = train_data[idx]
    print(f"=== SFT sample {idx} ===")
    print(f"instruction: {r['instruction'][:120]}...")
    inp = str(r.get("input", ""))
    print(f"input: {inp[:120]}...")
    out = r["output"]
    print(f"output ({len(out)} chars, first 300):")
    print(out[:300])
    print("...")
    print()
