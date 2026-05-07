"""
诊断脚本：DPO 训练后模型输出垃圾（逗号序列）的 root cause 分析

不修改任何项目代码，仅读取和对比。
用法: .\.venv\Scripts\python.exe tests\_diag_dpo_root_cause.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ============================================================
# 第一部分：DPO 数据集静态检查
# ============================================================
print("=" * 70)
print("PART 1: DPO 数据集静态检查")
print("=" * 70)

dpo_path = ROOT / "data" / "dpo_pairs.json"
if not dpo_path.exists():
    print(f"[FATAL] DPO 数据不存在: {dpo_path}")
    sys.exit(1)

dpo_pairs = []
with open(dpo_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dpo_pairs.append(json.loads(line))

print(f"[INFO] DPO pair 总数: {len(dpo_pairs)}")

# 检查 1: prompt 格式一致性
# DPO prompt 必须以 "Instruction:\n" 开头，以 "\n\n" 结尾
bad_prompt_format = 0
for i, p in enumerate(dpo_pairs):
    prompt = p.get("prompt", "")
    if not prompt.startswith("Instruction:\n"):
        bad_prompt_format += 1
        if bad_prompt_format <= 3:
            print(f"  [WARN] pair #{i}: prompt 不以 'Instruction:\\n' 开头: {prompt[:80]!r}...")
    if not prompt.endswith("\n\n"):
        bad_prompt_format += 1
        if bad_prompt_format <= 3:
            print(f"  [WARN] pair #{i}: prompt 不以 '\\n\\n' 结尾: ...{prompt[-40:]!r}")

if bad_prompt_format == 0:
    print("[OK] 所有 DPO prompt 格式一致（Instruction:\\n...\\n\\nInput:\\n...\\n\\n）")

# 检查 2: chosen/rejected 是否通过了 ast.parse
chosen_syntax_errors = 0
rejected_syntax_errors = 0
chosen_vuln = 0  # chosen 不应该是脆弱的
rejected_safe = 0  # rejected 应该是脆弱的
chosen_eq_rejected = 0  # chosen == rejected

from dataset.adversarial import contains_vulnerable_sql_pattern

for i, p in enumerate(dpo_pairs):
    chosen = p.get("chosen", "")
    rejected = p.get("rejected", "")
    
    # AST check
    try:
        ast.parse(chosen)
    except SyntaxError:
        chosen_syntax_errors += 1
    
    try:
        ast.parse(rejected)
    except SyntaxError:
        rejected_syntax_errors += 1
    
    # Vulnerability check
    cv, _ = contains_vulnerable_sql_pattern(chosen)
    if cv:
        chosen_vuln += 1
    
    rv, _ = contains_vulnerable_sql_pattern(rejected)
    if not rv:
        rejected_safe += 1
    
    # Equality check
    if chosen.strip() == rejected.strip():
        chosen_eq_rejected += 1

print(f"[CHECK] chosen 含语法错误: {chosen_syntax_errors}/{len(dpo_pairs)}")
print(f"[CHECK] rejected 含语法错误: {rejected_syntax_errors}/{len(dpo_pairs)}")
print(f"[CHECK] chosen 命中脆弱模式 (不应发生): {chosen_vuln}/{len(dpo_pairs)}")
print(f"[CHECK] rejected 未命中脆弱模式 (不应发生): {rejected_safe}/{len(dpo_pairs)}")
print(f"[CHECK] chosen == rejected (无偏好信号): {chosen_eq_rejected}/{len(dpo_pairs)}")

# 检查 3: 攻击类型分布
from collections import Counter
attack_counter = Counter(p.get("attack_type") for p in dpo_pairs)
print(f"[DIST] attack_type 分布: {dict(attack_counter)}")

diff_counter = Counter(p.get("difficulty") for p in dpo_pairs)
print(f"[DIST] difficulty 分布: {dict(diff_counter)}")

task_counter = Counter(p.get("task_type") for p in dpo_pairs)
print(f"[DIST] task_type 分布: {dict(task_counter)}")

# ============================================================
# 第二部分：Token 长度检查（max_prompt_length=512, max_length=1024）
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Token 长度检查")
print("=" * 70)

try:
    from transformers import AutoTokenizer
    base_model = "bigcode/starcoder2-3b"
    print(f"[INFO] 加载 tokenizer: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    prompt_lens = []
    chosen_lens = []
    rejected_lens = []
    total_lens_chosen = []
    total_lens_rejected = []
    
    max_prompt = 512
    max_total = 1024
    truncated_prompts = 0
    truncated_total = 0
    
    for p in dpo_pairs:
        prompt_ids = tok(p["prompt"], add_special_tokens=True)["input_ids"]
        chosen_ids = tok(p["chosen"], add_special_tokens=False)["input_ids"]
        rejected_ids = tok(p["rejected"], add_special_tokens=False)["input_ids"]
        
        pl = len(prompt_ids)
        cl = len(chosen_ids)
        rl = len(rejected_ids)
        prompt_lens.append(pl)
        chosen_lens.append(cl)
        rejected_lens.append(rl)
        total_lens_chosen.append(pl + cl)
        total_lens_rejected.append(pl + rl)
        
        if pl > max_prompt:
            truncated_prompts += 1
        if pl + cl > max_total or pl + rl > max_total:
            truncated_total += 1
    
    print(f"[TOKEN] prompt 长度: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"mean={sum(prompt_lens)/len(prompt_lens):.1f}, median={sorted(prompt_lens)[len(prompt_lens)//2]}")
    print(f"[TOKEN] chosen 长度: min={min(chosen_lens)}, max={max(chosen_lens)}, "
          f"mean={sum(chosen_lens)/len(chosen_lens):.1f}")
    print(f"[TOKEN] rejected 长度: min={min(rejected_lens)}, max={max(rejected_lens)}, "
          f"mean={sum(rejected_lens)/len(rejected_lens):.1f}")
    print(f"[TOKEN] prompt+chosen 总长: min={min(total_lens_chosen)}, max={max(total_lens_chosen)}, "
          f"mean={sum(total_lens_chosen)/len(total_lens_chosen):.1f}")
    print(f"[TOKEN] prompt 超 max_prompt_length(512) 的样本数: {truncated_prompts}/{len(dpo_pairs)}")
    print(f"[TOKEN] 总长超 max_length(1024) 的样本数: {truncated_total}/{len(dpo_pairs)}")
    
    # 详细列出超长样本
    if truncated_prompts > 0:
        print(f"\n[WARN] 以下 DPO pair 的 prompt 超过 512 tokens（会被截断，丢失前缀信息）：")
        for i, pl in enumerate(prompt_lens):
            if pl > 512:
                p = dpo_pairs[i]
                print(f"  pair #{i}: prompt_len={pl}, attack={p.get('attack_type')}, "
                      f"difficulty={p.get('difficulty')}, task={p.get('task_type')}")
                print(f"    prompt[:100]: {p['prompt'][:100]!r}...")

except ImportError as e:
    print(f"[SKIP] 无法加载 tokenizer: {e}")
except Exception as e:
    print(f"[ERROR] Token 分析失败: {e}")

# ============================================================
# 第三部分：chosen/rejected 内容质量抽查
# ============================================================
print("\n" + "=" * 70)
print("PART 3: DPO 对内容质量抽查（前 5 对）")
print("=" * 70)

for i in range(min(5, len(dpo_pairs))):
    p = dpo_pairs[i]
    print(f"\n--- DPO pair #{i} ---")
    print(f"  attack_type: {p.get('attack_type')}, difficulty: {p.get('difficulty')}, task: {p.get('task_type')}")
    print(f"  prompt (last 150 chars): ...{p['prompt'][-150:]!r}")
    print(f"  chosen: {p['chosen'][:200]!r}...")
    print(f"  rejected: {p['rejected'][:200]!r}...")

# ============================================================
# 第四部分：检查 chosen 中的代码与 prompt 的 driver/framework 一致性
# ============================================================
print("\n" + "=" * 70)
print("PART 4: chosen 代码 driver 与 prompt/attack_type 的一致性检查")
print("=" * 70)

driver_mismatch = 0
for i, p in enumerate(dpo_pairs):
    chosen = p.get("chosen", "")
    prompt = p.get("prompt", "")
    attack = p.get("attack_type", "")
    chosen_fw = p.get("chosen_framework", "unknown")
    
    # 检查 prompt 中提到的 driver
    prompt_has_pymysql = "pymysql" in prompt.lower()
    prompt_has_sqlalchemy = "sqlalchemy" in prompt.lower() or "ORM" in prompt
    
    # 检查 chosen 中实际使用的 driver
    chosen_has_pymysql = "import pymysql" in chosen
    chosen_has_sqlalchemy = "from sqlalchemy" in chosen or "import sqlalchemy" in chosen
    chosen_has_sqlite = "import sqlite3" in chosen
    
    if prompt_has_pymysql and not chosen_has_pymysql and not chosen_has_sqlite:
        driver_mismatch += 1
        if driver_mismatch <= 5:
            print(f"  [WARN] pair #{i}: prompt says pymysql but chosen uses {chosen_fw}")
            print(f"    chosen[:100]: {chosen[:100]!r}")

if driver_mismatch == 0:
    print("[OK] 所有 DPO pair 的 chosen 代码框架与 prompt 一致")
else:
    print(f"[WARN] {driver_mismatch}/{len(dpo_pairs)} DPO pair 的 chosen 框架与 prompt 不一致")

print("\n" + "=" * 70)
print("诊断完成。请根据以上输出判断 root cause。")
print("=" * 70)
