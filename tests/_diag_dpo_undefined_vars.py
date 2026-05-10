"""
诊断脚本 #3：检查 DPO 数据中 chosen 代码的未定义变量引用 bug
（_extract_likely_param 误识别 SQL 中间变量为用户参数）

用法: e:/graduation_proj_1/.venv/Scripts/python.exe tests/_diag_dpo_undefined_vars.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DPO_PATH = ROOT / "data" / "dpo_pairs.json"

def find_all_names(node: ast.AST) -> set[str]:
    """递归查找所有 Name 节点（包括函数调用中的）"""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
    return names

def find_defined_names(tree: ast.AST) -> set[str]:
    """查找所有被定义的名称（函数参数、赋值目标、import、函数定义）"""
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            for arg in node.args.args:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                defined.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                defined.add(name)
    # 内置函数（不需要定义）
    builtins = {"print", "len", "range", "int", "str", "float", "list", "dict",
                "set", "tuple", "bool", "type", "isinstance", "super", "enumerate",
                "zip", "map", "filter", "sorted", "reversed", "any", "all",
                "open", "input", "hasattr", "getattr", "setattr", "delattr",
                "abs", "min", "max", "sum", "round", "pow", "divmod",
                "ord", "chr", "repr", "format", "bytes", "bytearray",
                "Exception", "ValueError", "TypeError", "KeyError", "RuntimeError"}
    defined.update(builtins)
    defined.update({"self", "cls"})  # 隐式参数
    # 添加常见模块名
    defined.update({"pymysql", "sqlalchemy", "sqlite3", "text", "Session",
                    "DictCursor", "connections", "fetchall", "fetchone",
                    "fetchmany", "execute", "cursor", "conn", "cur", "session"})
    return defined

def check_undefined(code: str) -> tuple[bool, list[str]]:
    """检查代码中是否有未定义的名字引用。返回 (has_undefined, undefined_names)"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ["SYNTAX_ERROR"]
    
    defined = find_defined_names(tree)
    used = find_all_names(tree)
    
    # 排除 builtins 和属性访问
    undefined = used - defined
    # 再排除一些常见的
    undefined.discard("True")
    undefined.discard("False")
    undefined.discard("None")
    
    has_undef = len(undefined) > 0
    return has_undef, sorted(undefined)


def main():
    print("=" * 70)
    print("DPO Chosen 代码未定义变量检查")
    print("=" * 70)
    
    if not DPO_PATH.exists():
        print(f"[FATAL] DPO data not found: {DPO_PATH}")
        return
    
    dpo_pairs = []
    with open(DPO_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dpo_pairs.append(json.loads(line))
    
    print(f"[INFO] Total DPO pairs: {len(dpo_pairs)}")
    
    chosen_undef = []
    rejected_undef = []
    
    for i, p in enumerate(dpo_pairs):
        chosen = p.get("chosen", "")
        rejected = p.get("rejected", "")
        
        ch_undef, ch_names = check_undefined(chosen)
        rj_undef, rj_names = check_undefined(rejected)
        
        if ch_undef:
            chosen_undef.append({
                "index": i,
                "id": f"dpo_pair_{i}",
                "attack_type": p.get("attack_type"),
                "difficulty": p.get("difficulty"),
                "task_type": p.get("task_type"),
                "schema_table": p.get("schema_table"),
                "schema_column": p.get("schema_column"),
                "undefined_names": ch_names,
                "chosen_preview": chosen[:120],
                "rejected_preview": rejected[:120],
            })
        
        if rj_undef:
            rejected_undef.append({
                "index": i,
                "id": f"dpo_pair_{i}",
                "attack_type": p.get("attack_type"),
                "difficulty": p.get("difficulty"),
                "task_type": p.get("task_type"),
                "undefined_names": rj_names,
                "rejected_preview": rejected[:120],
            })
    
    print(f"\n[CRITICAL] chosen 含未定义变量: {len(chosen_undef)}/{len(dpo_pairs)}")
    print(f"[INFO] rejected 含未定义变量: {len(rejected_undef)}/{len(dpo_pairs)}")
    
    if chosen_undef:
        print("\n--- 详细列表（前 20 条） ---")
        for item in chosen_undef[:20]:
            print(f"\n  Pair #{item['index']}: attack={item['attack_type']}, "
                  f"diff={item['difficulty']}, task={item['task_type']}, "
                  f"table={item['schema_table']}, col={item['schema_column']}")
            print(f"    undefined: {item['undefined_names']}")
            print(f"    chosen: {item['chosen_preview']!r}...")
        
        # 统计分布
        from collections import Counter
        attack_dist = Counter(c["attack_type"] for c in chosen_undef)
        task_dist = Counter(c["task_type"] for c in chosen_undef)
        diff_dist = Counter(c["difficulty"] for c in chosen_undef)
        print(f"\n  按 attack_type: {dict(attack_dist)}")
        print(f"  按 task_type: {dict(task_dist)}")
        print(f"  按 difficulty: {dict(diff_dist)}")
    
    print("\n" + "=" * 70)
    print("诊断完成。")
    print("=" * 70)

if __name__ == "__main__":
    main()
