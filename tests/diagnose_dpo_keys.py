"""
诊断脚本：验证 StableDPOTrainer._prepare_dataset 输出格式
与 DataCollatorForPreference 期望格式是否匹配。

只读分析，不修改任何项目文件。
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import Dataset

# 1. 模拟 dpo_train.py 中的 load_dpo_dataset 加载逻辑
def load_dpo_dataset(path: Path) -> Dataset:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
                "attack_type": r.get("attack_type"),
                "difficulty": r.get("difficulty"),
                "task_type": r.get("task_type"),
                "expected_vulnerable": r.get("expected_vulnerable"),
                "schema_table": r.get("schema_table"),
                "schema_column": r.get("schema_column"),
                "chosen_framework": r.get("chosen_framework"),
            })
    return Dataset.from_list(rows)

# 2. 加载原始数据，查看其 key
dpo_path = ROOT / "data" / "dpo_pairs.json"
if not dpo_path.exists():
    print(f"[SKIP] DPO 数据文件不存在: {dpo_path}，尝试找其他文件...")
    # 尝试 dataset/dpo_train.jsonl
    alt_path = ROOT / "dataset" / "dpo_train.jsonl"
    if alt_path.exists():
        dpo_path = alt_path
    else:
        print("[ERROR] 找不到任何 DPO 数据文件")
        sys.exit(1)

print(f"[INFO] 使用数据文件: {dpo_path}")
ds = load_dpo_dataset(dpo_path)
print(f"[INFO] 原始数据集样本数: {len(ds)}")
print(f"[INFO] 原始数据集 keys: {list(ds[0].keys())}")

# 3. 查看原始数据集第 0 条的格式
sample = ds[0]
print(f"\n=== 原始数据集样本 (index 0) ===")
for k, v in sample.items():
    val_str = str(v)
    if len(val_str) > 100:
        val_str = val_str[:100] + "..."
    print(f"  {k}: {val_str}")

# 4. 模拟 tokenizer（不实际加载模型，只用星号代替）
print(f"\n=== 分析 _prepare_dataset 输出字段要求 ===")
print("""
DataCollatorForPreference.torch_call() 需要的 keys:
  - "prompt_ids"   (list of int token ids for the prompt)
  - "chosen_ids"   (list of int token ids for the chosen completion ONLY)
  - "rejected_ids" (list of int token ids for the rejected completion ONLY)

StableDPOTrainer._prepare_dataset() 实际输出的 keys:
  - "chosen_input_ids"      ← prompt + chosen 拼接
  - "chosen_attention_mask"
  - "chosen_labels"         ← -100 * len(prompt) + chosen_ids
  - "rejected_input_ids"    ← prompt + rejected 拼接
  - "rejected_attention_mask"
  - "rejected_labels"       ← -100 * len(prompt) + rejected_ids
  - "prompt_input_ids"      ← 仅 prompt 的 token ids
  - "prompt_attention_mask"
""")

print("=" * 60)
print("结论：字段名不匹配！")
print("=" * 60)
print("DataCollatorForPreference 需要: prompt_ids, chosen_ids, rejected_ids")
print("_prepare_dataset 输出:   prompt_input_ids, chosen_input_ids, rejected_input_ids")
print()
print("这就是 KeyError: 'prompt_ids' 的直接原因。")
print("当 _precompute_ref_logps 用 DataLoader(data_collator=DataCollatorForPreference)")
print("遍历 _prepare_dataset 的输出时，collator 找不到 'prompt_ids' 字段。")
