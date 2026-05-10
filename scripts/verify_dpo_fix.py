"""
快速验证脚本：仅运行 build_dpo_pairs 检查 DPO fallback 计数。
不修改任何数据文件。
"""
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.generate_expanded_dataset import build_dpo_pairs

# 加载现有训练数据
train_path = PROJECT_ROOT / "data" / "train_expanded.json"
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print(f"加载训练数据: {len(train_data)} 条")

# 运行 build_dpo_pairs（seed 固定）
rng = random.Random(42)
dpo_pairs = build_dpo_pairs(train_data, rng)

print(f"生成 DPO 对: {len(dpo_pairs)} 条")

# 统计 attack_type 分布
from collections import Counter
atk_counts = Counter(p["attack_type"] for p in dpo_pairs)
print(f"\nDPO 对 attack_type 分布:")
for atk, cnt in atk_counts.most_common():
    print(f"  {atk}: {cnt}")

print(f"\n✅ 验证完成。请检查终端输出中是否还有 [DPO fallback] 日志。")
