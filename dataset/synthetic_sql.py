"""合成 SQL 注入安全相关样本（本科规模可复现）。"""
from __future__ import annotations

import random
from typing import Any

DBS = ["sqlite3", "pymysql", "psycopg2"]
TABLES = ["users", "accounts", "employees", "orders"]
FIELDS = ["username", "email", "name", "phone"]


def _vuln_snippet(db: str, table: str, field: str) -> str:
    if db == "sqlite3":
        return f"""import sqlite3

def query_user(conn, value):
    sql = "SELECT * FROM {table} WHERE {field} = '" + value + "'"
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()
"""
    if db == "pymysql":
        return f"""import pymysql

def query_user(conn, value):
    sql = "SELECT * FROM {table} WHERE {field} = '" + value + "'"
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()
"""
    return f"""import psycopg2

def query_user(conn, value):
    sql = "SELECT * FROM {table} WHERE {field} = '" + value + "'"
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()
"""


def _safe_snippet(db: str, table: str, field: str) -> str:
    if db == "sqlite3":
        return f"""import sqlite3

def query_user(conn, value):
    sql = "SELECT * FROM {table} WHERE {field} = ?"
    cur = conn.cursor()
    cur.execute(sql, (value,))
    return cur.fetchall()
"""
    return f"""def query_user(conn, value):
    sql = "SELECT * FROM {table} WHERE {field} = %s"
    with conn.cursor() as cur:
        cur.execute(sql, (value,))
        return cur.fetchall()
"""


def gen_prompt(db: str, table: str, field: str) -> str:
    return (
        f"Write a Python function query_user(conn, value) that queries table {table} "
        f"by {field} using driver {db}. Use parameterized queries to avoid SQL injection. "
        f"Return fetched rows."
    )


def build_synthetic_splits(
    train_n: int,
    val_n: int,
    eval_prompts_n: int,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    train_sft: list[dict[str, Any]] = []
    val_sft: list[dict[str, Any]] = []
    train_dpo: list[dict[str, Any]] = []
    eval_prompts: list[dict[str, Any]] = []

    for i in range(train_n + val_n):
        db = random.choice(DBS)
        table = random.choice(TABLES)
        field = random.choice(FIELDS)
        p = gen_prompt(db, table, field)
        safe = _safe_snippet(db, table, field)
        vuln = _vuln_snippet(db, table, field)
        row_sft = {"id": i, "prompt": p, "completion": safe, "meta": {"db": db}}
        row_dpo = {"id": i, "prompt": p, "chosen": safe, "rejected": vuln}
        if i < train_n:
            train_sft.append(row_sft)
            train_dpo.append(row_dpo)
        else:
            val_sft.append(row_sft)

    for j in range(eval_prompts_n):
        db = random.choice(DBS)
        table = random.choice(TABLES)
        field = random.choice(FIELDS)
        p = gen_prompt(db, table, field)
        eval_prompts.append({"id": j, "prompt": p, "meta": {"db": db, "table": table}})

    return {
        "train_sft": train_sft,
        "val_sft": val_sft,
        "train_dpo": train_dpo,
        "eval_prompts": eval_prompts,
    }
