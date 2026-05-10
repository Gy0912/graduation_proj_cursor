"""
DPO fallback 诊断脚本
=====================
目的：找出为什么 AST+regex 策略对某些 (attack, difficulty, table, col) 
组合会"耗尽"（返回 None），导致回退到 _dispatch_vulnerable_aligned。

此脚本不修改任何项目文件，仅做只读分析。
"""

import ast
import json
import re
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path

# 添加项目根路径（脚本在 scripts/ 子目录下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 导入项目内部函数 ──
from dataset.generate_expanded_dataset import (
    _ast_find_execute_calls,
    _ast_find_sql_defs,
    _ast_find_indirect_sql,
    _ast_surgical_replace,
    _ast_surgical_variant,
    _try_variant_sqlalchemy,
    _try_variant_sqlite_pymysql_percent,
    _try_variant_indirect_full_query,
    _vulnerable_variant_from_chosen,
    _extract_imports,
    _extract_main_func_info,
    _detect_driver_from_code,
    extract_code_only_completion,
)
from dataset.adversarial import contains_vulnerable_sql_pattern


def load_train_data(path: str = None) -> list[dict]:
    """加载训练数据。"""
    if path is None:
        path = PROJECT_ROOT / "data" / "train_expanded.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_ast_steps(chosen_body: str, attack: str, table: str = "", col: str = "") -> dict:
    """
    逐步追踪 _ast_surgical_variant 的内部执行过程，返回每步的诊断信息。
    """
    result = {
        "chosen_preview": chosen_body[:300],
        "attack": attack,
        "steps": [],
        "final_result": None,
    }

    # Step 0: 检查 chosen 是否已含脆弱模式（不应发生）
    is_vuln, vuln_patterns = contains_vulnerable_sql_pattern(chosen_body)
    result["chosen_is_vuln"] = is_vuln
    result["chosen_vuln_patterns"] = vuln_patterns

    # Step 1: AST parse
    try:
        tree = ast.parse(chosen_body)
        result["steps"].append({"step": "ast_parse", "ok": True})
    except SyntaxError as e:
        result["steps"].append({"step": "ast_parse", "ok": False, "error": str(e)})
        return result

    source_lines = chosen_body.splitlines(True)

    # Step 2: 找 execute 调用
    exec_calls = _ast_find_execute_calls(tree, source_lines)
    result["steps"].append({
        "step": "find_execute_calls",
        "ok": len(exec_calls) > 0,
        "count": len(exec_calls),
        "details": [
            {
                "line_no": ec["line_no"],
                "sql_var_name": ec.get("sql_var_name"),
                "sql_str_direct": ec.get("sql_str_direct"),
                "param_var": ec.get("param_var"),
                "param_kind": ec.get("param_kind"),
                "is_sqlalchemy": ec.get("is_sqlalchemy"),
                "has_return_wrapper": ec.get("has_return_wrapper"),
            }
            for ec in exec_calls
        ],
    })

    if not exec_calls:
        result["final_result"] = "FAIL: no execute() calls found"
        return result

    # Step 3: 找 SQL 定义
    sql_defs = _ast_find_sql_defs(tree)
    result["steps"].append({
        "step": "find_sql_defs",
        "ok": len(sql_defs) > 0,
        "count": len(sql_defs),
        "vars": list(sql_defs.keys()),
        "details": [
            {"var": k, "sql_str": v["sql_str"], "is_text_wrapped": v["is_text_wrapped"]}
            for k, v in sql_defs.items()
        ],
    })

    # Step 4: 找间接 SQL
    indirect_sql = _ast_find_indirect_sql(tree)
    result["steps"].append({
        "step": "find_indirect_sql",
        "ok": indirect_sql is not None,
        "details": (
            {
                "func_name": indirect_sql["func_name"],
                "sql_str": indirect_sql["sql_str"],
            }
            if indirect_sql
            else None
        ),
    })

    # Step 5: 逐个尝试 execute call 的手术替换
    for i, exec_info in enumerate(exec_calls):
        sql_var = exec_info.get("sql_var_name")
        sql_def = sql_defs.get(sql_var) if sql_var else None
        has_direct_sql = exec_info.get("sql_str_direct") is not None

        replace_attempt = {
            "exec_index": i,
            "sql_var": sql_var,
            "has_sql_def": sql_def is not None,
            "has_indirect_sql": indirect_sql is not None,
            "has_direct_sql": has_direct_sql,
        }

        if sql_def is None and indirect_sql is None and not has_direct_sql:
            replace_attempt["fail_reason"] = (
                f"SQL var '{sql_var}' not found in sql_defs (keys: {list(sql_defs.keys())}), "
                f"no indirect_sql, and no direct SQL string in execute()"
            )
            result["steps"].append({"step": f"surgical_replace[{i}]", **replace_attempt})
            continue

        surgical_result = _ast_surgical_replace(
            source_lines, exec_info, sql_def, indirect_sql, attack,
            table=table, col=col,
        )
        if surgical_result is None:
            replace_attempt["fail_reason"] = "_ast_surgical_replace returned None"
            result["steps"].append({"step": f"surgical_replace[{i}]", **replace_attempt})
            continue

        # 检查结果语法
        try:
            ast.parse(surgical_result)
        except SyntaxError as e:
            replace_attempt["fail_reason"] = f"surgical result has SyntaxError: {e}"
            result["steps"].append({"step": f"surgical_replace[{i}]", **replace_attempt})
            continue

        # 检查结果是否脆弱
        is_vuln, vuln_pats = contains_vulnerable_sql_pattern(surgical_result)
        replace_attempt["is_vuln"] = is_vuln
        replace_attempt["vuln_patterns"] = vuln_pats
        replace_attempt["result_preview"] = surgical_result[:300]

        if not is_vuln:
            replace_attempt["fail_reason"] = "surgical result is NOT vulnerable per patterns"
            result["steps"].append({"step": f"surgical_replace[{i}]", **replace_attempt})
            continue

        replace_attempt["ok"] = True
        result["steps"].append({"step": f"surgical_replace[{i}]", **replace_attempt})
        result["final_result"] = "AST SURGICAL SUCCESS"
        return result

    # 所有 execute call 都失败
    result["final_result"] = "FAIL: all execute() calls exhausted in AST surgical"
    return result


def analyze_regex_steps(chosen_body: str, attack: str) -> dict:
    """追踪三个正则策略的匹配结果。"""
    rng = random.Random(42)
    result = {}

    # sqlalchemy 策略
    r1 = _try_variant_sqlalchemy(chosen_body, attack, rng)
    result["sqlalchemy"] = {
        "ok": r1 is not None,
        "preview": (r1[:200] if r1 else None),
    }
    if r1:
        try:
            ast.parse(r1)
            is_v, pats = contains_vulnerable_sql_pattern(r1)
            result["sqlalchemy"]["syntax_ok"] = True
            result["sqlalchemy"]["is_vuln"] = is_v
            result["sqlalchemy"]["vuln_patterns"] = pats
        except SyntaxError as e:
            result["sqlalchemy"]["syntax_ok"] = False
            result["sqlalchemy"]["syntax_error"] = str(e)

    # pymysql/sqlite 策略
    r2 = _try_variant_sqlite_pymysql_percent(chosen_body, attack, rng)
    result["pymysql_percent"] = {
        "ok": r2 is not None,
        "preview": (r2[:200] if r2 else None),
    }
    if r2:
        try:
            ast.parse(r2)
            is_v, pats = contains_vulnerable_sql_pattern(r2)
            result["pymysql_percent"]["syntax_ok"] = True
            result["pymysql_percent"]["is_vuln"] = is_v
            result["pymysql_percent"]["vuln_patterns"] = pats
        except SyntaxError as e:
            result["pymysql_percent"]["syntax_ok"] = False
            result["pymysql_percent"]["syntax_error"] = str(e)

    # indirect 策略
    r3 = _try_variant_indirect_full_query(chosen_body, attack, rng)
    result["indirect_full_query"] = {
        "ok": r3 is not None,
        "preview": (r3[:200] if r3 else None),
    }
    if r3:
        try:
            ast.parse(r3)
            is_v, pats = contains_vulnerable_sql_pattern(r3)
            result["indirect_full_query"]["syntax_ok"] = True
            result["indirect_full_query"]["is_vuln"] = is_v
            result["indirect_full_query"]["vuln_patterns"] = pats
        except SyntaxError as e:
            result["indirect_full_query"]["syntax_ok"] = False
            result["indirect_full_query"]["syntax_error"] = str(e)

    return result


def analyze_chosen_code_patterns(chosen_body: str) -> dict:
    """分析 chosen 代码的结构特征，帮助理解为何策略失败。"""
    info = {}

    # 检测 driver
    info["driver"] = _detect_driver_from_code(chosen_body)

    # 检测函数结构
    func_info = _extract_main_func_info(chosen_body)
    info["func_info"] = func_info

    # 检测 imports
    info["imports"] = _extract_imports(chosen_body)

    # 检测 execute 调用模式
    info["has_session_execute"] = "session.execute" in chosen_body
    info["has_cur_execute"] = "cur.execute" in chosen_body
    info["has_conn_execute"] = "conn.execute" in chosen_body

    # 检测 SQL 定义模式
    info["has_text_call"] = "text(" in chosen_body
    info["has_full_query"] = "def _full_query" in chosen_body
    info["has_percent_s"] = "%s" in chosen_body
    info["has_question_mark"] = "?" in chosen_body
    info["has_stmt_var"] = bool(re.search(r'^\s*stmt\s*=', chosen_body, re.MULTILINE))
    info["has_sql_var"] = bool(re.search(r'^\s*sql\s*=', chosen_body, re.MULTILINE))

    # 检测是否有 SELECT 语句
    info["has_select"] = bool(re.search(r'SELECT\s+\*', chosen_body, re.IGNORECASE))

    # 检测 SQL 字符串内容（text() 中的内容）
    text_matches = re.findall(r'text\("([^"]*)"\)', chosen_body)
    info["text_contents"] = text_matches

    # 检测 execute 调用的完整形式
    exec_matches = re.findall(
        r'(?:session\.|cur\.|conn\.)?execute\s*\([^)]+\)',
        chosen_body,
    )
    info["execute_calls"] = exec_matches

    return info


def main():
    print("=" * 80)
    print("DPO FALLBACK 诊断工具")
    print("=" * 80)

    # 加载数据
    data = load_train_data()
    print(f"\n加载训练数据: {len(data)} 条记录")

    # 筛选 expected_vulnerable == True 的记录（这些会触发 DPO 生成）
    dpo_rows = [r for r in data if r.get("expected_vulnerable") is True]
    print(f"其中 expected_vulnerable=True: {len(dpo_rows)} 条")

    # 按 attack_type 分组统计
    attack_counts = Counter(r.get("attack_type") for r in dpo_rows)
    print(f"\n按 attack_type 分布:")
    for atk, cnt in attack_counts.most_common():
        print(f"  {atk}: {cnt}")

    # 按 (attack_type, difficulty) 统计
    combo_counts = Counter(
        (r.get("attack_type"), r.get("difficulty")) for r in dpo_rows
    )
    print(f"\n按 (attack_type, difficulty) 分布:")
    for (atk, diff), cnt in sorted(combo_counts.items()):
        print(f"  ({atk}, {diff}): {cnt}")

    # ── 核心诊断：对每种 attack_type 的样本进行逐步追踪 ──
    print("\n" + "=" * 80)
    print("逐步诊断：每种 attack_type × difficulty 的 chosen 代码为何策略失败")
    print("=" * 80)

    # 收集所有失败案例
    rng = random.Random(42)
    failure_details = []  # (attack, difficulty, table, col, reason_summary, chosen_preview)

    # 按 attack+difficulty 分组，每组取最多 2 个样本分析
    grouped = defaultdict(list)
    for r in dpo_rows:
        key = (r.get("attack_type"), r.get("difficulty"))
        grouped[key].append(r)

    for (atk, diff), rows in sorted(grouped.items()):
        # 每组只分析前 2 个
        for r in rows[:2]:
            chosen_src = str(r.get("output", "")).strip()
            chosen_body = extract_code_only_completion(chosen_src)
            if not chosen_body:
                continue

            table = r.get("schema_table", "?")
            col = r.get("schema_column", "?")

            # 运行 _vulnerable_variant_from_chosen 看结果（传递 table/col）
            variant = _vulnerable_variant_from_chosen(
                chosen_body, atk, diff, rng,
                table=table, col=col,
            )

            if variant is not None:
                # 成功！不需要分析
                continue

            # 失败了，深入分析
            print(f"\n{'─' * 70}")
            print(f"❌ FALLBACK: attack={atk}, difficulty={diff}, table={table}, col={col}")
            print(f"{'─' * 70}")

            # 分析代码结构
            code_info = analyze_chosen_code_patterns(chosen_body)
            print(f"  Driver: {code_info['driver']}")
            print(f"  has_session_execute: {code_info['has_session_execute']}")
            print(f"  has_cur_execute: {code_info['has_cur_execute']}")
            print(f"  has_text_call: {code_info['has_text_call']}")
            print(f"  has_full_query: {code_info['has_full_query']}")
            print(f"  has_percent_s: {code_info['has_percent_s']}")
            print(f"  has_stmt_var: {code_info['has_stmt_var']}")
            print(f"  has_sql_var: {code_info['has_sql_var']}")
            print(f"  text_contents: {code_info['text_contents']}")
            print(f"  execute_calls: {code_info['execute_calls']}")

            # 追踪 AST 策略各步
            ast_analysis = analyze_ast_steps(chosen_body, atk, table=table, col=col)
            print(f"\n  [AST 策略最终结果]: {ast_analysis['final_result']}")

            for step in ast_analysis["steps"]:
                step_name = step["step"]
                if step_name == "ast_parse":
                    print(f"    Step ast_parse: {'✅' if step['ok'] else '❌'}")
                elif step_name == "find_execute_calls":
                    print(f"    Step find_execute_calls: {'✅' if step['ok'] else '❌'} ({step['count']} calls)")
                    for d in step.get("details", []):
                        print(f"      line={d['line_no']} sql_var={d['sql_var_name']} "
                              f"sql_direct={d['sql_str_direct']} param={d['param_var']} "
                              f"kind={d['param_kind']} sqlalchemy={d['is_sqlalchemy']}")
                elif step_name == "find_sql_defs":
                    print(f"    Step find_sql_defs: {'✅' if step['ok'] else '❌'} "
                          f"({step['count']} defs: {step['vars']})")
                    for d in step.get("details", []):
                        print(f"      var={d['var']} sql_str={d['sql_str']!r} "
                              f"text_wrapped={d['is_text_wrapped']}")
                elif step_name == "find_indirect_sql":
                    details = step.get("details")
                    print(f"    Step find_indirect_sql: {'✅' if step['ok'] else '❌'}"
                          f"{' — ' + str(details) if details else ''}")
                elif step_name.startswith("surgical_replace"):
                    ok = step.get("ok", False)
                    reason = step.get("fail_reason", "")
                    print(f"    Step {step_name}: {'✅' if ok else '❌'} "
                          f"sql_var={step.get('sql_var')} "
                          f"has_def={step.get('has_sql_def')} "
                          f"has_indirect={step.get('has_indirect_sql')} "
                          f"has_direct={step.get('has_direct_sql')}")
                    if reason:
                        print(f"      → {reason}")

            # 追踪正则策略
            regex_analysis = analyze_regex_steps(chosen_body, atk)
            print(f"\n  [正则策略结果]:")
            for strategy, info in regex_analysis.items():
                status = "✅" if info["ok"] else "❌"
                extra = ""
                if info["ok"]:
                    extra = f" syntax_ok={info.get('syntax_ok')} is_vuln={info.get('is_vuln')}"
                print(f"    {strategy}: {status}{extra}")

            # 显示实际代码
            print(f"\n  [Chosen 代码]:")
            for i, line in enumerate(chosen_body.splitlines()[:20], 1):
                print(f"    {i:3d}| {line}")

            # 记录失败详情
            failure_details.append({
                "attack": atk,
                "difficulty": diff,
                "table": table,
                "col": col,
                "ast_result": ast_analysis["final_result"],
                "driver": code_info["driver"],
                "has_execute": code_info["has_cur_execute"] or code_info["has_session_execute"],
                "has_sql_def": len(code_info["text_contents"]) > 0,
                "execute_calls": code_info["execute_calls"],
            })

    # ── 汇总报告 ──
    print("\n\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)

    print(f"\n总失败案例（已分析）: {len(failure_details)}")

    # 按 AST 失败原因分类
    reason_counts = Counter(f["ast_result"] for f in failure_details)
    print(f"\nAST 策略失败原因分布:")
    for reason, cnt in reason_counts.most_common():
        print(f"  {reason}: {cnt}")

    # 按 driver 分布
    driver_counts = Counter(f["driver"] for f in failure_details)
    print(f"\nDriver 分布:")
    for d, cnt in driver_counts.most_common():
        print(f"  {d}: {cnt}")

    # 按 (attack, driver) 分布
    combo2 = Counter((f["attack"], f["driver"]) for f in failure_details)
    print(f"\n(attack, driver) 分布:")
    for (a, d), cnt in sorted(combo2.items()):
        print(f"  ({a}, {d}): {cnt}")

    # 关键发现
    print(f"\n{'─' * 70}")
    print("关键发现:")
    print(f"{'─' * 70}")

    no_execute = [f for f in failure_details if not f["has_execute"]]
    if no_execute:
        print(f"  1. {len(no_execute)} 个案例的 chosen 代码中完全没有 execute() 调用")
        for f in no_execute[:5]:
            print(f"     - ({f['attack']}, {f['difficulty']}, {f['table']}, {f['col']})")
        print(f"     → 这些代码可能使用了 ORM .filter() / .query() 而非 execute()")

    has_execute_but_no_def = [
        f for f in failure_details
        if f["has_execute"] and "FAIL: all execute" in str(f["ast_result"])
    ]
    if has_execute_but_no_def:
        print(f"  2. {len(has_execute_but_no_def)} 个案例有 execute() 但 SQL 定义未被识别")
        for f in has_execute_but_no_def[:5]:
            print(f"     - ({f['attack']}, {f['difficulty']}, {f['table']}, {f['col']}) "
                  f"driver={f['driver']} calls={f['execute_calls']}")

    print(f"\n诊断完成。请查看上述输出定位根因。")


if __name__ == "__main__":
    main()
