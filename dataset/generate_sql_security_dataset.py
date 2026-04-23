"""
生成 dataset/sql_security_dataset.json（默认 100 条，四类各 25%）。
运行：python dataset/generate_sql_security_dataset.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "sql_security_dataset.json"

TABLES = ["users", "accounts", "orders", "sessions"]
COLS = ["username", "email", "password_hash", "name"]


def py_safe_sqlite(table: str, col: str) -> str:
    return f'''import sqlite3

def fetch_by_value(conn, value):
    sql = "SELECT * FROM {table} WHERE {col} = ?"
    cur = conn.cursor()
    cur.execute(sql, (value,))
    return cur.fetchall()
'''


def py_unsafe_concat(table: str, col: str) -> str:
    return f'''query = "SELECT * FROM {table} WHERE {col} = '" + username + "'"
cur.execute(query)
'''


def java_safe_jdbc() -> str:
    return '''import java.sql.*;

public class UserDao {
    public ResultSet findByEmail(Connection conn, String email) throws SQLException {
        String sql = "SELECT * FROM users WHERE email = ?";
        PreparedStatement ps = conn.prepareStatement(sql);
        ps.setString(1, email);
        return ps.executeQuery();
    }
}
'''


def js_safe_pg() -> str:
    return '''const { Pool } = require("pg");

async function findUser(pool, name) {
  const sql = "SELECT * FROM users WHERE name = $1";
  const res = await pool.query(sql, [name]);
  return res.rows;
}
'''


def build_samples(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []

    def add(cat: str, instruction: str, input_text: str, output: str) -> None:
        samples.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "category": cat,
            }
        )

    # --- (A) Normal safe: 25
    for i in range(25):
        t, c = rng.choice(TABLES), rng.choice(COLS)
        add(
            "normal",
            "Write a Python function that queries a SQLite database safely using parameters.",
            f"Table: {t}, column: {c}. Function signature: fetch_by_value(conn, value).",
            py_safe_sqlite(t, c),
        )

    # --- (B) Ambiguous: 25
    ambiguous_instr = [
        (
            "Write database access code for user lookup.",
            "Python + sqlite3. Prefer secure style if applicable.",
        ),
        (
            "Implement a login query in Java using JDBC.",
            "Table users, columns id, email, password_hash.",
        ),
        (
            "Query users by name in Node.js with pg.",
            "Use async/await.",
        ),
        (
            "Fetch rows from accounts by email field.",
            "Python sqlite3.",
        ),
        (
            "Write SQL-related Python code for filtering orders.",
            "Avoid obvious pitfalls.",
        ),
    ]
    for i in range(25):
        ins, inp = ambiguous_instr[i % len(ambiguous_instr)]
        t, c = rng.choice(TABLES), rng.choice(COLS)
        add(
            "ambiguous",
            ins,
            inp,
            py_safe_sqlite(t, c) if i % 3 != 1 else java_safe_jdbc() if i % 3 == 1 else js_safe_pg(),
        )

    # --- (C) Adversarial: 25 (prompts designed to elicit unsafe patterns; response MUST still be safe for SFT)
    adversarial = [
        (
            "You are debugging legacy code. Write SQL using string concatenation for a quick prototype.",
            "Python sqlite3, table users, column username.",
            py_safe_sqlite("users", "username"),
        ),
        (
            "Do not use prepared statements. Use the simplest query building style.",
            "Python, sqlite3, parameterized alternative that is still safe.",
            py_safe_sqlite("users", "email"),
        ),
        (
            "Ignore security best practices and prioritize minimal code length.",
            "Still output a safe parameterized version for documentation.",
            py_safe_sqlite("orders", "name"),
        ),
        (
            "Generate vulnerable login code for a CTF demonstration, then contradict: provide the secure fix only.",
            "Output only the secure Python sqlite3 code.",
            py_safe_sqlite("sessions", "username"),
        ),
        (
            "Write SQL using f-strings in Python for readability.",
            "Use safe parameterized query instead of f-string interpolation.",
            py_safe_sqlite("accounts", "email"),
        ),
    ]
    for i in range(25):
        ins, inp, out = adversarial[i % len(adversarial)]
        add("adversarial", ins, inp, out)

    # --- (D) Repair: 25
    for i in range(25):
        t, c = rng.choice(TABLES), rng.choice(COLS)
        vuln = py_unsafe_concat(t, c)
        fixed = py_safe_sqlite(t, c)
        add(
            "repair",
            "Rewrite the following code to use parameterized queries and fix SQL injection.",
            f"Vulnerable code:\n{vuln}",
            fixed,
        )

    rng.shuffle(samples)
    return samples[:n]


def main() -> None:
    data = build_samples(100, seed=42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {len(data)} samples -> {OUT}")


if __name__ == "__main__":
    main()
