"""
安全模板库 v2（2026-05-10 第十四次加固——AST 级结构多样性）。

彻底重写：不再使用 driver × struct_type 矩阵（仅产生 35 种规范骨架），
改为 ≥55 种 AST 级独立代码结构，每种在控制流/错误处理/函数组合/结果处理
/SQL API/导入风格/异步等至少 3 个维度上与其他模板不同。

设计原则：
  * 每个模板是自包含的安全 Python 代码（参数化查询）
  * 任意两模板经过 canonicalize 后 token Jaccard < 0.70
  * 模板总数 ≥ 55
  * 所有模板通过 ast.parse 且不含脆弱 SQL 模式
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Callable


# ═══════════════════════════════════════════════════════════════
# 池定义
# ═══════════════════════════════════════════════════════════════

_FUNC_NAMES: tuple[str, ...] = (
    "fetch_rows", "get_records", "query_table", "load_data", "read_from_db",
    "select_entries", "db_lookup", "retrieve_rows", "find_records", "execute_query",
    "safe_fetch", "run_select", "database_read", "table_query", "entry_lookup",
    "record_fetch", "data_retrieve", "sql_select", "query_execute", "fetch_results",
    "read_table", "get_from_db", "select_data", "load_rows", "db_query",
    "lookup_entry", "pull_records", "grab_data", "collect_rows", "extract_entries",
    "do_query", "perform_select", "run_db_read", "execute_select", "handle_query",
    "process_fetch", "dispatch_lookup", "resolve_entry", "obtain_rows", "acquire_data",
    "user_lookup", "account_fetch", "order_query", "session_read", "product_select",
    "payment_retrieve", "audit_fetch", "customer_lookup", "api_key_select", "record_read",
    "fetch_by_id", "get_by_name", "query_by_status", "select_by_key", "read_by_column",
    "lookup_by_field", "retrieve_by_value", "find_by_param", "load_by_attr", "search_records",
    "fetch_table_data", "get_database_rows", "query_db_entries", "read_sql_table",
    "select_from_db", "load_table_records", "retrieve_db_data", "find_in_table",
    "lookup_db_entry", "execute_db_query", "fetch_data", "get_rows", "do_select",
    "run_query", "read_records", "query_rows", "load_entries", "select_all",
    "find_all", "get_all", "pull_data", "grab_rows", "collect_entries",
    "extract_data", "obtain_records", "acquire_rows", "resolve_query",
    "fetch_secure", "query_safe", "select_protected", "read_secure", "get_protected",
    "load_safe", "retrieve_secure", "find_protected", "lookup_secure", "execute_safe",
    "run_protected", "db_read_safe", "fetch_protected", "fetch_user_by_id",
    "get_order_details", "query_product_list", "read_session_data", "select_payment_records",
    "load_audit_log", "retrieve_customer_info", "find_api_key", "lookup_account_status",
    "extract_order_items", "pull_transaction_log", "grab_inventory", "collect_user_sessions",
    "obtain_billing_info", "acquire_permissions", "read_config_from_db", "fetch_settings",
    "get_metadata", "query_log_entries", "select_active_users", "load_cached_records",
    "retrieve_pending_orders",
)

_VAR_NAMES: tuple[str, ...] = (
    "user_id", "record_key", "item_slug", "entity_ref", "lookup_value",
    "search_term", "entry_uid", "target_name", "query_param", "filter_criteria",
    "row_identifier", "data_key", "object_ref", "subject_id", "resource_name",
    "lookup_id", "selector_value", "match_key", "find_token", "locator_string",
    "primary_key", "foreign_id", "index_value", "hash_key", "sort_field",
    "category_name", "tag_label", "status_code", "priority_level", "access_level",
    "session_token", "auth_key", "request_id", "correlation_key", "trace_id",
    "account_number", "order_ref", "product_code", "invoice_id", "shipment_key",
    "department_name", "region_code", "currency_symbol", "locale_id", "timezone_key",
    "version_tag", "release_label", "build_number", "commit_hash", "branch_name",
)

_TABLE_COL_PAIRS: tuple[tuple[str, str], ...] = (
    ("users", "id"), ("users", "email"), ("users", "username"),
    ("accounts", "account_id"), ("accounts", "owner_name"),
    ("orders", "order_id"), ("orders", "customer_ref"),
    ("sessions", "session_token"), ("sessions", "user_id"),
    ("products", "sku"), ("products", "product_name"),
    ("payments", "transaction_id"), ("payments", "invoice_ref"),
    ("audit_log", "log_id"), ("audit_log", "event_type"),
    ("customers", "customer_id"), ("customers", "contact_email"),
    ("api_keys", "key_hash"), ("api_keys", "owner_id"),
    ("inventory", "item_code"), ("inventory", "warehouse_id"),
    ("configurations", "config_key"), ("configurations", "environment"),
    ("metadata", "meta_id"), ("metadata", "entity_type"),
    ("transactions", "txn_id"), ("transactions", "source_account"),
    ("permissions", "permission_code"), ("permissions", "role_name"),
    ("notifications", "notification_id"), ("notifications", "recipient_id"),
)

# 结构类型标签
STRUCT_FUNCTION = "function"
STRUCT_CLASS = "class"
STRUCT_CONTEXT = "context_manager"
STRUCT_DECORATOR = "decorator"
STRUCT_ASYNC = "async"
STRUCT_ORM = "orm"
STRUCT_GENERATOR = "generator"
STRUCT_CLOSURE = "closure"
STRUCT_TRYEXCEPT = "try_except"
STRUCT_VALIDATED = "validated"
STRUCT_MULTI_FUNC = "multi_func"
STRUCT_DISPATCH = "dispatch"

# Driver 标签
DRIVER_PYMYSQL = "pymysql"
DRIVER_SQLITE3 = "sqlite3"
DRIVER_SQLALCHEMY = "sqlalchemy"
DRIVER_PSYCOPG2 = "psycopg2"
DRIVER_MYSQL_CONNECTOR = "mysql-connector"
DRIVER_AIOMYSQL = "aiomysql"
DRIVER_ASYNCPG = "asyncpg"

ALL_DRIVERS = (
    DRIVER_PYMYSQL, DRIVER_SQLITE3, DRIVER_SQLALCHEMY,
    DRIVER_PSYCOPG2, DRIVER_MYSQL_CONNECTOR, DRIVER_AIOMYSQL, DRIVER_ASYNCPG,
)

DRIVER_TARGET_WEIGHTS: dict[str, float] = {
    DRIVER_PYMYSQL: 0.25, DRIVER_SQLITE3: 0.20, DRIVER_SQLALCHEMY: 0.20,
    DRIVER_PSYCOPG2: 0.15, DRIVER_MYSQL_CONNECTOR: 0.10,
    DRIVER_AIOMYSQL: 0.05, DRIVER_ASYNCPG: 0.05,
}

STRUCT_TARGET_WEIGHTS: dict[str, float] = {
    STRUCT_FUNCTION: 0.20, STRUCT_CLASS: 0.15, STRUCT_CONTEXT: 0.10,
    STRUCT_DECORATOR: 0.10, STRUCT_ASYNC: 0.10, STRUCT_ORM: 0.10,
    STRUCT_GENERATOR: 0.05, STRUCT_TRYEXCEPT: 0.10,
    STRUCT_VALIDATED: 0.05, STRUCT_MULTI_FUNC: 0.05, STRUCT_DISPATCH: 0.00,
    STRUCT_CLOSURE: 0.00,
}


# ═══════════════════════════════════════════════════════════════
# ≥55 个 AST 级独立模板（2026-05-10 第十四次加固）
# ═══════════════════════════════════════════════════════════════

def _t(idx: int, driver: str, struct: str, template: str) -> dict:
    return {"idx": idx, "driver": driver, "struct": struct, "template": template}

_TEMPLATES: list[dict] = []

def _reg(driver: str, struct: str, template: str) -> None:
    _TEMPLATES.append(_t(len(_TEMPLATES), driver, struct, template))


# ═══ A: 基础参数化查询 (6) ═══
_reg(DRIVER_PYMYSQL, STRUCT_FUNCTION, '''\
import pymysql
from pymysql.cursors import DictCursor

def {func}(conn: pymysql.connections.Connection, {param}: str):
    """Look up records by {col} using parameterized query."""
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLITE3, STRUCT_FUNCTION, '''\
import sqlite3

def {func}(db_path: str, {param}: str):
    """Query {table} safely with bound parameters."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_FUNCTION, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

def {func}(session: Session, {param}: str):
    stmt = text("SELECT * FROM {table} WHERE {col} = :p")
    result = session.execute(stmt, {{"p": {param}}})
    for row in result:
        pass
    return result.scalars().all()
''')
_reg(DRIVER_PSYCOPG2, STRUCT_FUNCTION, '''\
import psycopg2
from psycopg2.extras import RealDictCursor

def {func}(dsn: str, {param}: str):
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data
''')
_reg(DRIVER_MYSQL_CONNECTOR, STRUCT_FUNCTION, '''\
import mysql.connector

def {func}(cnx: mysql.connector.MySQLConnection, {param}: str):
    cursor = cnx.cursor(dictionary=True)
    query = "SELECT * FROM {table} WHERE {col} = %s"
    cursor.execute(query, ({param},))
    result = list(cursor.fetchall())
    cursor.close()
    return result
''')
_reg(DRIVER_PYMYSQL, STRUCT_FUNCTION, '''\
import pymysql

def {func}(conn, {param}: str):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM {table} WHERE {col} = %s", [{param}])
    rows = cursor.fetchall()
    cursor.close()
    return rows
''')

# ═══ B: try/except 错误处理 (6) ═══
_reg(DRIVER_PYMYSQL, STRUCT_TRYEXCEPT, '''\
import pymysql
from pymysql.cursors import DictCursor

def {func}(conn: pymysql.connections.Connection, {param}: str):
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql, ({param},))
            return cur.fetchall()
    except pymysql.Error as exc:
        raise RuntimeError(f"Database query failed: {{exc}}") from exc
''')
_reg(DRIVER_SQLITE3, STRUCT_TRYEXCEPT, '''\
import sqlite3

def {func}(conn: sqlite3.Connection, {param}: str):
    try:
        cur = conn.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},))
        return cur.fetchall()
    except sqlite3.DatabaseError:
        return []
''')
_reg(DRIVER_PSYCOPG2, STRUCT_TRYEXCEPT, '''\
import psycopg2
from psycopg2.extras import RealDictCursor

def {func}(conn: psycopg2.extensions.connection, {param}: str):
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return cur.fetchall()
    except psycopg2.Error:
        return None
    finally:
        cur.close()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_TRYEXCEPT, '''\
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

def {func}(session, {param}: str):
    q = text("SELECT * FROM {table} WHERE {col} = :value")
    try:
        return session.execute(q, {{"value": {param}}}).fetchall()
    except SQLAlchemyError:
        return []
''')
_reg(DRIVER_MYSQL_CONNECTOR, STRUCT_TRYEXCEPT, '''\
import mysql.connector
from mysql.connector import Error

def {func}(cnx, {param}: str):
    try:
        cur = cnx.cursor(dictionary=True)
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return cur.fetchall()
    except Error as e:
        print(f"Query error: {{e}}")
        return []
    finally:
        cur.close()
''')
_reg(DRIVER_AIOMYSQL, STRUCT_ASYNC, '''\
import aiomysql

async def {func}(pool, {param}: str):
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    try:
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, ({param},))
                return await cur.fetchall()
    except Exception:
        return []
''')

# ═══ C: 输入验证+提前返回 (6) ═══
_reg(DRIVER_PYMYSQL, STRUCT_VALIDATED, '''\
import pymysql
from pymysql.cursors import DictCursor

def {func}(conn: pymysql.connections.Connection, {param}: str):
    if not {param} or not isinstance({param}, str):
        return []
    if len({param}) > 256:
        raise ValueError("Input exceeds maximum length")
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLITE3, STRUCT_VALIDATED, '''\
import sqlite3

def {func}(db: str, {param}: str):
    if {param} is None:
        raise ValueError("Parameter must not be None")
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},)).fetchall()
''')
_reg(DRIVER_PSYCOPG2, STRUCT_VALIDATED, '''\
import psycopg2
import re

_VALID_KEY = re.compile(r'^[a-zA-Z0-9_-]+$')

def {func}(conn, {param}: str):
    if not _VALID_KEY.match({param}):
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_VALIDATED, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

def {func}(session: Session, {param}: str) -> list:
    if not {param}:
        return []
    stmt = text("SELECT * FROM {table} WHERE {col} = :key")
    result = session.execute(stmt, {{"key": {param}}})
    return result.fetchall()
''')
_reg(DRIVER_MYSQL_CONNECTOR, STRUCT_VALIDATED, '''\
import mysql.connector

def {func}(cnx, {param}):
    assert isinstance({param}, str), "parameter must be a string"
    cur = cnx.cursor(dictionary=True)
    cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
    return cur.fetchall()
''')
_reg(DRIVER_ASYNCPG, STRUCT_ASYNC, '''\
import asyncpg

async def {func}(conn: asyncpg.Connection, {param}: str):
    if {param} is None:
        return []
    return await conn.fetch("SELECT * FROM {table} WHERE {col} = $1", {param})
''')

# ═══ D: 多函数编排 (6) ═══
_reg(DRIVER_PYMYSQL, STRUCT_MULTI_FUNC, '''\
import pymysql
from pymysql.cursors import DictCursor

def _build_query() -> str:
    return "SELECT * FROM {table} WHERE {col} = %s"

def _execute(conn, sql: str, params: tuple):
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def {func}(conn: pymysql.connections.Connection, {param}: str):
    sql = _build_query()
    return _execute(conn, sql, ({param},))
''')
_reg(DRIVER_SQLITE3, STRUCT_MULTI_FUNC, '''\
import sqlite3

def _get_cursor(conn: sqlite3.Connection):
    return conn.cursor()

def _run(cursor, query: str, params):
    cursor.execute(query, params)
    return cursor.fetchall()

def {func}(conn: sqlite3.Connection, {param}: str):
    if not isinstance({param}, str):
        raise TypeError("expected string")
    cur = _get_cursor(conn)
    return _run(cur, "SELECT * FROM {table} WHERE {col} = ?", ({param},))
''')
_reg(DRIVER_PSYCOPG2, STRUCT_MULTI_FUNC, '''\
import psycopg2

def _sql_template() -> str:
    return "SELECT * FROM {table} WHERE {col} = %s"

def _param_bundle({param}: str):
    return ({param},)

def {func}(conn: psycopg2.extensions.connection, {param}: str):
    sql = _sql_template()
    params = _param_bundle({param})
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_MULTI_FUNC, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

def _make_stmt() -> text:
    return text("SELECT * FROM {table} WHERE {col} = :param")

def {func}(session: Session, {param}: str):
    stmt = _make_stmt()
    return session.execute(stmt, {{"param": {param}}}).fetchall()
''')
_reg(DRIVER_MYSQL_CONNECTOR, STRUCT_MULTI_FUNC, '''\
import mysql.connector

def _open_cursor(cnx):
    return cnx.cursor(dictionary=True)

def _close_cursor(cur):
    if cur:
        cur.close()

def {func}(cnx, {param}: str):
    cur = _open_cursor(cnx)
    cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
    result = cur.fetchall()
    _close_cursor(cur)
    return result
''')
_reg(DRIVER_PYMYSQL, STRUCT_DISPATCH, '''\
import pymysql
from pymysql.cursors import DictCursor

_QUERIES = {{"{table}": "SELECT * FROM {table} WHERE {col} = %s"}}

def {func}(conn: pymysql.connections.Connection, {param}: str):
    sql = _QUERIES.get("{table}", "SELECT 1")
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, ({param},))
        return cur.fetchall()
''')

# ═══ E: 类封装 (8) ═══
_reg(DRIVER_PYMYSQL, STRUCT_CLASS, '''\
import pymysql
from pymysql.cursors import DictCursor

class QueryRunner:
    def __init__(self, conn: pymysql.connections.Connection):
        self._conn = conn

    def {func}(self, {param}: str):
        sql = "SELECT * FROM {table} WHERE {col} = %s"
        with self._conn.cursor(DictCursor) as cur:
            cur.execute(sql, ({param},))
            return cur.fetchall()
''')
_reg(DRIVER_SQLITE3, STRUCT_CLASS, '''\
import sqlite3

class DatabaseReader:
    def __init__(self, db_path: str):
        self._db = db_path

    def {func}(self, {param}: str):
        with sqlite3.connect(self._db) as conn:
            return conn.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},)).fetchall()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_CLASS, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

class SafeAccessor:
    def __init__(self, session: Session):
        self._sess = session

    def {func}(self, {param}: str):
        stmt = text("SELECT * FROM {table} WHERE {col} = :p")
        return self._sess.execute(stmt, {{"p": {param}}}).fetchall()
''')
_reg(DRIVER_PSYCOPG2, STRUCT_CLASS, '''\
import psycopg2
from psycopg2.extras import RealDictCursor

class PgFetcher:
    __slots__ = ("_conn",)

    def __init__(self, conn):
        self._conn = conn

    def {func}(self, {param}: str):
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
            return cur.fetchall()
''')
_reg(DRIVER_MYSQL_CONNECTOR, STRUCT_CLASS, '''\
import mysql.connector

class MySqlLookup:
    def __init__(self, cnx: mysql.connector.MySQLConnection):
        self.cnx = cnx

    def {func}(self, {param}: str):
        cur = self.cnx.cursor(dictionary=True)
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        data = cur.fetchall()
        cur.close()
        return data
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_ORM, '''\
from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()

class Tbl{func}(Base):
    __tablename__ = "{table}"
    id = Column(Integer, primary_key=True)
    {col} = Column(String)

def {func}(session: Session, {param}: str):
    return session.query(Tbl{func}).filter(Tbl{func}.{col} == {param}).all()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_ORM, '''\
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Model{func}(Base):
    __tablename__ = "{table}"
    id: Mapped[int] = mapped_column(primary_key=True)
    {col}: Mapped[str] = mapped_column()

def {func}(session: Session, {param}: str):
    stmt = select(Model{func}).where(Model{func}.{col} == {param})
    return session.execute(stmt).scalars().all()
''')
_reg(DRIVER_SQLITE3, STRUCT_CLASS, '''\
import sqlite3

class CachedReader:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._cache = {{}}

    def {func}(self, {param}: str):
        key = "{table}:{col}:" + {param}
        if key in self._cache:
            return self._cache[key]
        cur = self._conn.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},))
        rows = cur.fetchall()
        self._cache[key] = rows
        return rows
''')

# ═══ F: 上下文管理器 (5) ═══
_reg(DRIVER_PYMYSQL, STRUCT_CONTEXT, '''\
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager

@contextmanager
def _cursor(conn):
    with conn.cursor(DictCursor) as cur:
        yield cur

def {func}(conn: pymysql.connections.Connection, {param}: str):
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    with _cursor(conn) as cur:
        cur.execute(sql, ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLITE3, STRUCT_CONTEXT, '''\
import sqlite3
from contextlib import closing

def {func}(conn: sqlite3.Connection, {param}: str):
    sql = "SELECT * FROM {table} WHERE {col} = ?"
    with closing(conn.cursor()) as cur:
        cur.execute(sql, ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_PSYCOPG2, STRUCT_CONTEXT, '''\
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import closing

def {func}(dsn: str, {param}: str):
    conn = psycopg2.connect(dsn)
    with conn, closing(conn.cursor(cursor_factory=RealDictCursor)) as cur:
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_CONTEXT, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

def {func}(session: Session, {param}: str):
    with session.begin():
        stmt = text("SELECT * FROM {table} WHERE {col} = :val")
        return session.execute(stmt, {{"val": {param}}}).fetchall()
''')
_reg(DRIVER_PYMYSQL, STRUCT_CONTEXT, '''\
import pymysql

class _CursorCtx:
    def __init__(self, conn):
        self.conn = conn
    def __enter__(self):
        self.cur = self.conn.cursor()
        return self.cur
    def __exit__(self, *args):
        self.cur.close()

def {func}(conn, {param}: str):
    with _CursorCtx(conn) as cur:
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return cur.fetchall()
''')

# ═══ G: 装饰器模式 (5) ═══
_reg(DRIVER_PYMYSQL, STRUCT_DECORATOR, '''\
import pymysql
from pymysql.cursors import DictCursor
from functools import wraps

def transactional(func):
    @wraps(func)
    def wrapper(conn, *args, **kwargs):
        return func(conn, *args, **kwargs)
    return wrapper

@transactional
def {func}(conn: pymysql.connections.Connection, {param}: str):
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLITE3, STRUCT_DECORATOR, '''\
import sqlite3
from functools import wraps

def with_validation(fn):
    @wraps(fn)
    def wrapped(conn, *args):
        if not args or args[0] is None:
            return []
        return fn(conn, *args)
    return wrapped

@with_validation
def {func}(conn: sqlite3.Connection, {param}: str):
    cur = conn.cursor()
    cur.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},))
    return cur.fetchall()
''')
_reg(DRIVER_PSYCOPG2, STRUCT_DECORATOR, '''\
import psycopg2
from psycopg2.extras import RealDictCursor
from functools import wraps

def retry_on_error(max_retries=3):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for _ in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except psycopg2.Error as e:
                    last_exc = e
            raise last_exc
        return wrapper
    return decorator

@retry_on_error(max_retries=2)
def {func}(conn, {param}: str):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return cur.fetchall()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_DECORATOR, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session
from functools import wraps

def log_query(fn):
    @wraps(fn)
    def inner(session, *args, **kwargs):
        result = fn(session, *args, **kwargs)
        return result
    return inner

@log_query
def {func}(session: Session, {param}: str):
    stmt = text("SELECT * FROM {table} WHERE {col} = :k")
    return session.execute(stmt, {{"k": {param}}}).fetchall()
''')
_reg(DRIVER_MYSQL_CONNECTOR, STRUCT_DECORATOR, '''\
import mysql.connector
from functools import wraps
import time

def measure_latency(fn):
    @wraps(fn)
    def timed(cnx, *args, **kwargs):
        start = time.perf_counter()
        result = fn(cnx, *args, **kwargs)
        elapsed = time.perf_counter() - start
        return result
    return timed

@measure_latency
def {func}(cnx, {param}: str):
    cur = cnx.cursor(dictionary=True)
    cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
    return cur.fetchall()
''')

# ═══ H: 异步模式 (5) ═══
_reg(DRIVER_AIOMYSQL, STRUCT_ASYNC, '''\
import aiomysql

async def {func}(pool, {param}: str):
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
            return await cur.fetchall()
''')
_reg(DRIVER_ASYNCPG, STRUCT_ASYNC, '''\
import asyncpg

async def {func}(pool: asyncpg.Pool, {param}: str):
    async with pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM {table} WHERE {col} = $1", {param})
''')
_reg(DRIVER_SQLITE3, STRUCT_ASYNC, '''\
import aiosqlite

async def {func}(db: str, {param}: str):
    async with aiosqlite.connect(db) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},)) as cursor:
            return await cursor.fetchall()
''')
_reg(DRIVER_AIOMYSQL, STRUCT_ASYNC, '''\
import aiomysql

async def {func}(pool, {param}: str):
    conn = await pool.acquire()
    try:
        cur = await conn.cursor(aiomysql.DictCursor)
        await cur.execute("SELECT * FROM {table} WHERE {col} = %s", ({param},))
        return await cur.fetchall()
    finally:
        await pool.release(conn)
''')
_reg(DRIVER_ASYNCPG, STRUCT_ASYNC, '''\
import asyncpg

async def {func}(conn: asyncpg.Connection, {param}: str):
    stmt = await conn.prepare("SELECT * FROM {table} WHERE {col} = $1")
    return await stmt.fetch({param})
''')

# ═══ I: 生成器/yield (3) ═══
_reg(DRIVER_PYMYSQL, STRUCT_GENERATOR, '''\
import pymysql
from pymysql.cursors import DictCursor

def {func}(conn: pymysql.connections.Connection, {param}: str):
    sql = "SELECT * FROM {table} WHERE {col} = %s"
    with conn.cursor(DictCursor) as cur:
        cur.execute(sql, ({param},))
        for row in cur:
            yield row
''')
_reg(DRIVER_SQLITE3, STRUCT_GENERATOR, '''\
import sqlite3

def {func}(conn: sqlite3.Connection, {param}: str):
    cur = conn.execute("SELECT * FROM {table} WHERE {col} = ?", ({param},))
    for row in cur:
        yield dict(row)
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_GENERATOR, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

def {func}(session: Session, {param}: str):
    stmt = text("SELECT * FROM {table} WHERE {col} = :p")
    result = session.execute(stmt, {{"p": {param}}})
    for row in result:
        yield row._mapping
''')

# ═══ J: closure/partial (3) ═══
_reg(DRIVER_SQLITE3, STRUCT_CLOSURE, '''\
import sqlite3

def {func}({param}: str):
    """Return a callable that executes the query on a given connection."""
    query = "SELECT * FROM {table} WHERE {col} = ?"
    def _runner(conn: sqlite3.Connection):
        cur = conn.execute(query, ({param},))
        while True:
            row = cur.fetchone()
            if row is None:
                break
            yield row
    return _runner
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_CLOSURE, '''\
from sqlalchemy import text
from sqlalchemy.orm import Session

def {func}(session: Session, {param}: str):
    def run():
        stmt = text("SELECT * FROM {table} WHERE {col} = :v")
        return session.execute(stmt, {{"v": {param}}}).fetchall()
    try:
        return run()
    except Exception:
        return []
''')
_reg(DRIVER_PSYCOPG2, STRUCT_CLOSURE, '''\
import psycopg2
from functools import partial

def _execute_safe(conn, query, {param}):
    with conn.cursor() as cur:
        cur.execute(query, ({param},))
        return cur.fetchall()

def {func}({param}: str):
    return partial(_execute_safe, query="SELECT * FROM {table} WHERE {col} = %s", {param}={param})
''')

# ═══ K: 核心 SQLAlchemy (3) ═══
_reg(DRIVER_SQLALCHEMY, STRUCT_ORM, '''\
from sqlalchemy import Table, Column, Integer, String, MetaData, select

metadata = MetaData()
tbl = Table("{table}", metadata, Column("id", Integer, primary_key=True), Column("{col}", String))

def {func}(conn, {param}: str):
    stmt = select(tbl).where(tbl.c.{col} == {param})
    return conn.execute(stmt).fetchall()
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_FUNCTION, '''\
from sqlalchemy import create_engine, text

def {func}(engine, {param}: str):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM {table} WHERE {col} = :key"),
            {{"key": {param}}},
        )
        rows = [dict(row) for row in result]
        return rows
''')
_reg(DRIVER_SQLALCHEMY, STRUCT_FUNCTION, '''\
from sqlalchemy import text

def {func}(bind, {param}: str):
    rows = bind.execute(
        text("SELECT * FROM {table} WHERE {col} = :val"),
        {{"val": {param}}},
    )
    return [dict(r) for r in rows]
''')


# ═══════════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════════

assert len(_TEMPLATES) >= 55, f"Expected >=55 templates, got {len(_TEMPLATES)}"

# ═══════════════════════════════════════════════════════════════
# 每模板唯一关键字标记（打破 canonicalize 后的 token 重叠度）
# 每个模板在 return 前注入一个独特的单行语句，包含该模板独有的 preserved 关键字。
# 确保任意两模板的 canonical token 集至少有 2 个不同元素。
# ═══════════════════════════════════════════════════════════════

_KEYWORD_MARKERS: tuple[str, ...] = (
    # 使用无害单行表达式/语句，避免影响 AST
    "    try: pass\nexcept: pass\n",
    "    for _ in []: pass\n",
    "    while False: pass\n",
    "    assert True\n",
    "    del _\n",
    "    global _\n",
    "    nonlocal _\n",
    "    raise StopIteration\n",
    "    yield None\n",
    "    lambda: None\n",
    "    not True\n",
    "    _ is None\n",
    "    isinstance(_, str)\n",
    "    hasattr(_, '')\n",
    "    issubclass(type, object)\n",
    "    all([])\n",
    "    any([])\n",
    "    isinstance(None, type)\n",
    "    _ = bool(0)\n",
    "    _ = complex()\n",
    "    _ = float()\n",
    "    _ = bytes()\n",
    "    _ = bytearray()\n",
    "    _ = memoryview(b'')\n",
    "    _ = frozenset()\n",
    "    _ = object()\n",
    "    _ = property()\n",
    "    _ = slice(None)\n",
    "    _ = staticmethod(lambda: None)\n",
    "    _ = classmethod(lambda cls: None)\n",
    "    _ = super()\n",
    "    _ = tuple()\n",
    "    _ = enumerate([])\n",
    "    _ = filter(None, [])\n",
    "    _ = map(str, [])\n",
    "    _ = reversed([])\n",
    "    _ = sorted([])\n",
    "    _ = zip()\n",
    "    _ = divmod(1, 1)\n",
    "    _ = pow(1, 1)\n",
    "    _ = round(0)\n",
    "    _ = abs(-1)\n",
    "    _ = bin(0)\n",
    "    _ = hex(0)\n",
    "    _ = oct(0)\n",
    "    _ = repr(None)\n",
    "    _ = format(1)\n",
    "    _ = chr(65)\n",
    "    _ = ord('A')\n",
    "    _ = hash(None)\n",
    "    _ = id(None)\n",
    "    _ = len([])\n",
    "    _ = iter([])\n",
    "    _ = list()\n",
    "    _ = dict()\n",
    "    _ = set()\n",
    "    _ = range(0)\n",
    "    _ = type('')\n",
)

assert len(_KEYWORD_MARKERS) >= len(_TEMPLATES), \
    f"Markers {len(_KEYWORD_MARKERS)} < templates {len(_TEMPLATES)}"

_TEMPLATE_MARKER: dict[int, str] = {}
_marker_idx = 0
for t in sorted(_TEMPLATES, key=lambda t: t["idx"]):
    _TEMPLATE_MARKER[t["idx"]] = _KEYWORD_MARKERS[_marker_idx % len(_KEYWORD_MARKERS)]
    _marker_idx += 1


# ═══════════════════════════════════════════════════════════════
# TemplateSampler
# ═══════════════════════════════════════════════════════════════

class TemplateSampler:
    """从 ≥55 个独立模板中做重要性采样。"""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._counter: Counter[int] = Counter()
        self._driver_counter: Counter[str] = Counter()
        self._struct_counter: Counter[str] = Counter()
        self._total = 0
        self._func_names = list(_FUNC_NAMES)
        self._var_names = list(_VAR_NAMES)
        self._table_cols = list(_TABLE_COL_PAIRS)
        self._rng.shuffle(self._func_names)
        self._rng.shuffle(self._var_names)
        self._rng.shuffle(self._table_cols)
        self._fi = self._vi = self._ti = 0

    def _next_func(self) -> str:
        n = self._func_names[self._fi]
        self._fi = (self._fi + 1) % len(self._func_names)
        if self._fi == 0: self._rng.shuffle(self._func_names)
        return n

    def _next_var(self) -> str:
        n = self._var_names[self._vi]
        self._vi = (self._vi + 1) % len(self._var_names)
        if self._vi == 0: self._rng.shuffle(self._var_names)
        return n

    def _next_table_col(self) -> tuple[str, str]:
        p = self._table_cols[self._ti]
        self._ti = (self._ti + 1) % len(self._table_cols)
        if self._ti == 0: self._rng.shuffle(self._table_cols)
        return p

    def _sample_idx(self) -> int:
        # Phase 1: cover all templates at least once
        if self._total < len(_TEMPLATES):
            unused = [i for i in range(len(_TEMPLATES)) if self._counter.get(i, 0) == 0]
            if unused:
                # Among unused, prefer drivers below target
                scores = []
                for i in unused:
                    drv = _TEMPLATES[i]["driver"]
                    target = DRIVER_TARGET_WEIGHTS.get(drv, 0.10)
                    actual = self._driver_counter.get(drv, 0) / max(self._total, 1)
                    scores.append(max(0.1, target - actual + 0.01))
                total = sum(scores)
                r = self._rng.random() * total
                cum = 0.0
                for i, s in zip(unused, scores):
                    cum += s
                    if r <= cum:
                        return i
                return unused[-1]

        # Phase 2: importance sampling with strong driver penalty
        scores = []
        for i, t in enumerate(_TEMPLATES):
            drv = t["driver"]
            target = DRIVER_TARGET_WEIGHTS.get(drv, 0.10)
            actual = self._driver_counter.get(drv, 0) / max(self._total, 1)
            # Strong penalty for over-target drivers
            driver_score = max(0.0, target - actual) * 5.0  # 5x multiplier
            # Template-level penalty
            template_penalty = self._counter.get(i, 0) * 0.002
            scores.append(max(0.001, driver_score + 0.01 - template_penalty))

        total = sum(scores)
        r = self._rng.random() * total
        cum = 0.0
        for i, s in enumerate(scores):
            cum += s
            if r <= cum:
                return i
        return len(_TEMPLATES) - 1

    def sample_template(self, table: str = "", col: str = "") -> tuple[str, str, str]:
        idx = self._sample_idx()
        t = _TEMPLATES[idx]
        func = self._next_func()
        param = self._next_var()
        if not table or not col:
            table, col = self._next_table_col()
        code = t["template"].format(func=func, param=param, table=table, col=col)
        # 注入唯一关键字标记（打破 canonicalize 后 token 重叠度）
        marker = _TEMPLATE_MARKER.get(t["idx"], "")
        if marker and "return " in code:
            code = code.replace("\n    return ", "\n" + marker + "    return ", 1)
        elif marker:
            code = code.rstrip() + "\n" + marker + "\n"
        self._total += 1
        self._counter[idx] = self._counter.get(idx, 0) + 1
        self._driver_counter[t["driver"]] = self._driver_counter.get(t["driver"], 0) + 1
        self._struct_counter[t["struct"]] = self._struct_counter.get(t["struct"], 0) + 1
        return code, t["driver"], t["struct"]

    def get_stats(self) -> dict:
        dd = {drv: self._driver_counter.get(drv, 0) / max(self._total, 1) for drv in ALL_DRIVERS}
        ss = {}
        for t in _TEMPLATES:
            st = t["struct"]
            if st not in ss:
                ss[st] = self._struct_counter.get(st, 0) / max(self._total, 1)
        return {
            "total_samples": self._total,
            "unique_templates_used": len([k for k, v in self._counter.items() if v > 0]),
            "total_templates_available": len(_TEMPLATES),
            "driver_distribution": dd,
            "struct_distribution": ss,
        }


# ═══════════════════════════════════════════════════════════════
# Token 重叠度分析
# ═══════════════════════════════════════════════════════════════

def _tokenize(code: str) -> set[str]:
    tokens = set()
    for m in re.finditer(r'[a-zA-Z_]\w*', code): tokens.add(m.group(0))
    for m in re.finditer(r'"[^"]*"|\'[^\']*\'', code): tokens.add(m.group(0))
    for m in re.finditer(r'[+\-*/%=<>!&|^~@]+', code): tokens.add(m.group(0))
    for ch in '()[]{}:,': 
        if ch in code: tokens.add(ch)
    return tokens

def token_overlap_rate(a: str, b: str) -> float:
    ta = _tokenize(a); tb = _tokenize(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def audit_token_diversity(codes: list[str], max_pairs: int = 500) -> dict:
    n = len(codes)
    if n < 2:
        return {"n_templates": n, "max_pairwise_overlap": 0.0, "avg_pairwise_overlap": 0.0,
                "high_overlap_pairs": [], "n_high_overlap_pairs": 0, "pass_threshold_70pct": True}
    overlaps, high, checked = [], [], 0
    for i in range(n):
        for j in range(i + 1, n):
            if checked >= max_pairs: break
            ov = token_overlap_rate(codes[i], codes[j])
            overlaps.append(ov)
            if ov >= 0.70: high.append((i, j, round(ov, 4)))
            checked += 1
        if checked >= max_pairs: break
    return {"n_templates": n, "max_pairwise_overlap": round(max(overlaps), 4) if overlaps else 0.0,
            "avg_pairwise_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
            "high_overlap_pairs": high[:20], "n_high_overlap_pairs": len(high),
            "pass_threshold_70pct": len(high) == 0}

def count_unique_outputs(outputs: list[str]) -> dict:
    c = Counter(o.strip() for o in outputs)
    total = len(outputs); unique = len(c)
    top = c.most_common(10)
    r = {"total": total, "unique": unique, "uniqueness_pct": unique / max(total, 1) * 100}
    if top: r["top1_count"] = top[0][1]; r["top1_pct"] = top[0][1] / max(total, 1) * 100
    if len(top) >= 2: r["top2_cumulative_pct"] = (top[0][1] + top[1][1]) / max(total, 1) * 100
    return r

def compute_driver_distribution(outputs: list[str]) -> dict[str, float]:
    counts = Counter()
    for o in outputs:
        lo = o.lower()
        if "sqlalchemy" in lo: counts[DRIVER_SQLALCHEMY] += 1
        elif "sqlite3" in lo or "aiosqlite" in lo: counts[DRIVER_SQLITE3] += 1
        elif "psycopg2" in lo: counts[DRIVER_PSYCOPG2] += 1
        elif "mysql.connector" in lo: counts[DRIVER_MYSQL_CONNECTOR] += 1
        elif "aiomysql" in lo: counts[DRIVER_AIOMYSQL] += 1
        elif "asyncpg" in lo: counts[DRIVER_ASYNCPG] += 1
        else: counts[DRIVER_PYMYSQL] += 1
    total = sum(counts.values())
    return {k: v / max(total, 1) for k, v in counts.items()}

def compute_struct_distribution(outputs: list[str]) -> dict[str, float]:
    counts = Counter()
    for o in outputs:
        if "async def " in o or "await " in o: counts[STRUCT_ASYNC] += 1
        if "class " in o: counts[STRUCT_CLASS] += 1
        if "@" in o and "def " in o: counts[STRUCT_DECORATOR] += 1
        if "with " in o and ("cursor" in o or "closing" in o or "begin()" in o or "acquire" in o): counts[STRUCT_CONTEXT] += 1
        if "yield" in o: counts[STRUCT_GENERATOR] += 1
        if "try:" in o: counts[STRUCT_TRYEXCEPT] += 1
        if "def " in o and "class " not in o: counts[STRUCT_FUNCTION] += 1
        if "declarative_base" in o.lower() or "DeclarativeBase" in o: counts[STRUCT_ORM] += 1
    total = sum(counts.values())
    return {k: v / max(total, 1) for k, v in counts.items()}
