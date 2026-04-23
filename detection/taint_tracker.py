"""
轻量动态污点追踪：TaintedStr + sqlite3 execute 探针，在受限 exec 环境中运行片段。
"""
from __future__ import annotations

import ast
import sqlite3
import types
from typing import Any

# ---------------------------------------------------------------------------
# Tainted string
# ---------------------------------------------------------------------------


class TaintedStr(str):
    """带污点标记的 str 子类；拼接 / 格式化时传播污点。"""

    tainted: bool

    def __new__(cls, value: str, tainted: bool = False) -> TaintedStr:
        obj = str.__new__(cls, value)
        obj.tainted = bool(tainted)
        return obj

    def __add__(self, other: Any) -> TaintedStr:
        if not isinstance(other, str):
            return NotImplemented  # type: ignore[return-value]
        res = str.__add__(self, other)
        ot = isinstance(other, TaintedStr) and other.tainted
        return TaintedStr(res, self.tainted or ot)

    def __radd__(self, other: Any) -> TaintedStr:
        if not isinstance(other, str):
            return NotImplemented  # type: ignore[return-value]
        res = str.__add__(other, self)
        return TaintedStr(res, self.tainted or (isinstance(other, TaintedStr) and other.tainted))

    def __mod__(self, other: Any) -> TaintedStr:
        res = str.__mod__(self, other)
        t = self.tainted
        if isinstance(other, tuple):
            for x in other:
                if isinstance(x, TaintedStr) and x.tainted:
                    t = True
                    break
        elif isinstance(other, dict):
            for x in other.values():
                if isinstance(x, TaintedStr) and x.tainted:
                    t = True
                    break
        elif isinstance(other, TaintedStr) and other.tainted:
            t = True
        return TaintedStr(res, t)

    def format(self, *args: Any, **kwargs: Any) -> TaintedStr:
        plain = str.format(self, *args, **kwargs)
        t = self.tainted
        for a in args:
            if isinstance(a, TaintedStr) and a.tainted:
                t = True
        for v in kwargs.values():
            if isinstance(v, TaintedStr) and v.tainted:
                t = True
        return TaintedStr(plain, t)


def taint_input(value: str) -> TaintedStr:
    return TaintedStr(value, True)


def _coerce_taint(val: Any) -> bool:
    return bool(getattr(val, "tainted", False))


def _t_fv_full(val: Any, conversion: int, spec_expr_result: str | None) -> TaintedStr:
    """模拟 FormattedValue：conversion 为 -1 或 ord('s'/'r'/'a')；spec 已为普通 str 或 None。"""
    t = _coerce_taint(val)
    v = val
    if conversion == ord("r"):
        v = repr(val)
    elif conversion == ord("s"):
        v = str(val)
    elif conversion == ord("a"):
        v = ascii(val)
    else:
        v = val
    spec = spec_expr_result or ""
    if spec:
        body = format(v, spec)
    else:
        body = str(v)
    return TaintedStr(body, t)


# ---------------------------------------------------------------------------
# f-string → TaintedStr 拼接链（避免 BUILD_STRING 丢失子类）
# ---------------------------------------------------------------------------


class _FStringToTaintChain(ast.NodeTransformer):
    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.expr:
        expr: ast.expr | None = None
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                piece = ast.Call(
                    func=ast.Name("TaintedStr", ast.Load()),
                    args=[ast.Constant(part.value), ast.Constant(False)],
                    keywords=[],
                )
            elif isinstance(part, ast.FormattedValue):
                val_expr = self.visit(part.value)
                conv = ast.Constant(part.conversion)
                if part.format_spec:
                    spec_inner = self.visit(part.format_spec)
                    spec_wrapped = ast.Call(
                        func=ast.Name("str", ast.Load()),
                        args=[spec_inner],
                        keywords=[],
                    )
                else:
                    spec_wrapped = ast.Constant(None)
                piece = ast.Call(
                    func=ast.Name("_t_fv_full", ast.Load()),
                    args=[val_expr, conv, spec_wrapped],
                    keywords=[],
                )
            else:
                continue
            if expr is None:
                expr = piece
            else:
                expr = ast.BinOp(left=expr, op=ast.Add(), right=piece)
        return expr if expr is not None else ast.Constant("")


def _transform_fstrings(source: str) -> str:
    tree = ast.parse(source)
    fixed = _FStringToTaintChain().visit(tree)
    ast.fix_missing_locations(fixed)
    return ast.unparse(fixed)


# ---------------------------------------------------------------------------
# sqlite3：沙箱模块（包装 connect/cursor/execute，避免修改不可变的 Connection 类型）
# ---------------------------------------------------------------------------

_flow_details: list[dict[str, Any]] = []
_SQLITE3_SANDBOX: types.ModuleType | None = None


def _query_is_tainted(sql: Any) -> bool:
    if isinstance(sql, TaintedStr):
        return bool(sql.tainted)
    return bool(getattr(sql, "tainted", False))


def _record_sql_sink(sql: Any, sink: str) -> None:
    if _query_is_tainted(sql):
        _flow_details.append({"sink": sink, "sql_preview": str(sql)[:200]})


def _build_sqlite3_sandbox() -> types.ModuleType:
    import sqlite3 as _real

    m = types.ModuleType("sqlite3")

    class _CursorWrapper:
        __slots__ = ("_c",)

        def __init__(self, c: sqlite3.Cursor) -> None:
            self._c = c

        def execute(self, sql: Any, parameters: Any = ()) -> Any:
            if _query_is_tainted(sql):
                _record_sql_sink(sql, "sqlite3.Cursor.execute")
                return None
            return self._c.execute(str(sql), parameters)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._c, name)

    class _ConnWrapper:
        __slots__ = ("_conn",)

        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: Any, parameters: Any = ()) -> Any:
            if _query_is_tainted(sql):
                _record_sql_sink(sql, "sqlite3.Connection.execute")
                return None
            return self._conn.execute(str(sql), parameters)

        def cursor(self) -> _CursorWrapper:
            return _CursorWrapper(self._conn.cursor())

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    def connect(*args: Any, **kwargs: Any) -> _ConnWrapper:
        return _ConnWrapper(_real.connect(*args, **kwargs))

    m.connect = connect
    for attr in ("Error", "OperationalError", "sqlite_version", "PARSE_DECLTYPES"):
        if hasattr(_real, attr):
            setattr(m, attr, getattr(_real, attr))
    m.__name__ = "sqlite3"
    return m


def _get_sqlite3_sandbox() -> types.ModuleType:
    global _SQLITE3_SANDBOX
    if _SQLITE3_SANDBOX is None:
        _SQLITE3_SANDBOX = _build_sqlite3_sandbox()
    return _SQLITE3_SANDBOX


# ---------------------------------------------------------------------------
# 受限执行环境
# ---------------------------------------------------------------------------

def _safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> types.ModuleType:
    if name == "sqlite3":
        return _get_sqlite3_sandbox()
    raise ImportError(f"module {name!r} is not allowed in taint sandbox")


def _safe_builtins() -> dict[str, Any]:
    import builtins

    names = (
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "chr",
        "dict",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "AttributeError",
        "KeyError",
        "IndexError",
        "StopIteration",
    )
    out: dict[str, Any] = {n: getattr(builtins, n) for n in names}
    out["__import__"] = _safe_import
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_taint_analysis(code: str) -> dict[str, Any]:
    """
    在受限 globals 下 exec 代码，检测污点是否进入 sqlite3 execute/sql。

    Returns
    -------
    dict
        is_vulnerable, taint_flows_detected, details, error
    """
    global _flow_details
    err: str | None = None
    try:
        transformed = _transform_fstrings(code)
    except SyntaxError as e:
        return {
            "is_vulnerable": False,
            "taint_flows_detected": 0,
            "details": [],
            "error": f"transform_syntax_error: {e}",
        }

    _flow_details.clear()
    try:
        g: dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "TaintedStr": TaintedStr,
            "taint_input": taint_input,
            "_t_fv_full": _t_fv_full,
            "sqlite3": _get_sqlite3_sandbox(),
            "input": lambda _prompt="": taint_input("__stdin__"),
        }
        exec(compile(transformed, "<taint_sandbox>", "exec"), g, g)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    n = len(_flow_details)
    return {
        "is_vulnerable": n > 0,
        "taint_flows_detected": n,
        "details": list(_flow_details),
        "error": err,
    }


__all__ = [
    "TaintedStr",
    "run_taint_analysis",
    "taint_input",
]
