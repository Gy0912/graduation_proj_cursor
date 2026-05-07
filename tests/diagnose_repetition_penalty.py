"""
诊断脚本：验证评测配置中 repetition_penalty 缺失导致贪心解码退化循环。

只读分析，不修改任何项目文件。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

print("=" * 60)
print("诊断：评测 generation 参数缺失导致贪心解码退化")
print("=" * 60)

# 1. 读取 default.yaml（修复后的源配置）
default_yaml = ROOT / "configs" / "default.yaml"
cfg_default = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
gen_default = cfg_default.get("generation", {})
print(f"\n[default.yaml] generation 参数:")
for k in ["max_new_tokens", "temperature", "top_p", "repetition_penalty", "no_repeat_ngram_size"]:
    print(f"  {k}: {gen_default.get(k, '<<MISSING>>')}")

# 2. 读取 default_run.yaml（实际运行时使用的配置）
default_run_yaml = ROOT / "configs" / "default_run.yaml"
if default_run_yaml.exists():
    cfg_run = yaml.safe_load(default_run_yaml.read_text(encoding="utf-8"))
    gen_run = cfg_run.get("generation", {})
    print(f"\n[default_run.yaml] generation 参数:")
    for k in ["max_new_tokens", "temperature", "top_p", "repetition_penalty", "no_repeat_ngram_size"]:
        val = gen_run.get(k)
        status = "<<MISSING!>>" if val is None else val
        print(f"  {k}: {status}")

# 3. 模拟 evaluate.py 的参数解析逻辑
gen_to_use = gen_run if default_run_yaml.exists() else gen_default
repetition_penalty = float(gen_to_use.get("repetition_penalty", 1.0))
no_repeat_ngram_size = int(gen_to_use.get("no_repeat_ngram_size", 0))
temperature = gen_to_use.get("temperature", 0)

print(f"\n=== 模拟 evaluate.py 参数解析 ===")
print(f"  gen.get('repetition_penalty', 1.0) = {repetition_penalty}")
print(f"  gen.get('no_repeat_ngram_size', 0) = {no_repeat_ngram_size}")
print(f"  temperature = {temperature}")

# 4. 模拟 model.generate() 实际传入的参数
actual_rp = repetition_penalty if repetition_penalty != 1.0 else None
actual_nrns = no_repeat_ngram_size if no_repeat_ngram_size > 0 else None
print(f"\n=== 实际传入 model.generate() 的参数 ===")
print(f"  repetition_penalty = {actual_rp}  ← {'❌ None（无重复惩罚）' if actual_rp is None else '✅ 有效'}")
print(f"  no_repeat_ngram_size = {actual_nrns}  ← {'❌ None（无 n-gram 去重）' if actual_nrns is None else '✅ 有效'}")
print(f"  do_sample = {temperature > 0}  ← {'❌ 贪心解码（最易退化）' if temperature == 0 else '✅ 采样解码'}")

print("\n" + "=" * 60)
print("结论：")
print("=" * 60)

if actual_rp is None and temperature == 0:
    print("""
⚠️  关键问题：default_run.yaml 缺少 repetition_penalty 参数！

default.yaml（2026-05-05 修复版）已添加：
    repetition_penalty: 1.05
    # 注释：贪心解码（temperature=0）无重复惩罚会导致 token 退化循环

但 default_run.yaml 可能来自旧版 default.yaml（修复前生成），
导致 evaluate.py 获取到 repetition_penalty=1.0（默认值），
进而在 model.generate() 中被过滤为 None。

复现路径：
  1. temperature=0 → do_sample=False（贪心解码）
  2. repetition_penalty=1.0 → 被过滤为 None（无重复惩罚）
  3. 贪心解码 + 无重复惩罚 → 模型一旦选了 return" 就永远选 return"
     → 输出变成几百行 return" + return" + ...
     → extract_python_code 无法解析 → invalid_extraction

这与 default.yaml 2026-05-05 修复注释描述的现象完全一致：
    「贪心解码（temperature=0）无重复惩罚会导致 token 退化循环。
      repetition_penalty=1.05 温和降低已出现 token 的概率，
      避免 return cur.fetchall() 无限重复。」
""")
else:
    print("repetition_penalty 正常。")

print(f"\n修复方法：重新生成 default_run.yaml")
print(f"  python scripts/prepare_default_run.py")
print(f"或手动在 default_run.yaml 的 generation 段添加：")
print(f"  repetition_penalty: 1.05")
