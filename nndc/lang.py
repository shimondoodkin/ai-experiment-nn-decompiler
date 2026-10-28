"""The program language: a restricted Python subset.

Programs are parsed with `ast` against an allowlist and run by a vectorized
interpreter. They are never eval'd, exec'd or imported.

Only stdlib + NumPy: this module is copied into blind folders.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LANG_VERSION = 1
FUNCTIONS = {"relu", "sigmoid", "tanh", "min", "max", "abs", "sum", "argmax"}

_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
           ast.Eq: np.equal, ast.NotEq: np.not_equal}
_UNARYOPS = (ast.USub, ast.UAdd, ast.Not)


class ProgramError(ValueError):
    pass


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # same formula as model.activate (not imported: fix folders copy lang.py without model.py)
    with np.errstate(under="ignore"):   # exp(-|z|) underflowing to 0 is exact enough
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def allowlist_summary() -> str:
    return (
        "exactly one function `def net(x0, x1, ...)` with the same arguments as dump.py; "
        "statements: assignment to a plain name, if/elif/else, return; "
        "expressions: numbers, True/False, + - *, division by a nonzero number constant, "
        "unary - and not, and/or, comparisons < <= > >= == != (chains allowed), "
        "`a if cond else b`, and the functions relu(x), sigmoid(x), tanh(x), min(...), max(...), abs(x), "
        "sum(a, b, ...) or sum((a, b, ...)), argmax(a, b, ...) (index of the first maximum); "
        "comparisons give 1/0. No imports, loops, other functions, attributes, ** or comprehensions."
    )


def _err(node: ast.AST | None, what: str) -> ProgramError:
    line = getattr(node, "lineno", None)
    prefix = f"line {line}: " if line else ""
    return ProgramError(f"{prefix}{what}")


def _not_allowed(node: ast.AST, at: ast.AST | None = None) -> ProgramError:
    # operator nodes (ast.Pow, ast.Lt, ...) are shared singletons without positions: report `at`'s line
    return _err(at or node, f"{type(node).__name__} is not allowed (see the allowed language)")


def _is_number_const(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)):
        v = float(node.operand.value)
        return -v if isinstance(node.op, ast.USub) else v
    return None


def _check_expr(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float, bool):
            raise _err(node, f"constant {node.value!r} is not allowed (numbers and True/False only)")
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise _not_allowed(node.op, node)
        if isinstance(node.op, ast.Div):
            c = _is_number_const(node.right)
            if c is None or c == 0:
                raise _err(node, "division is allowed only by a nonzero number constant")
        _check_expr(node.left)
        _check_expr(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARYOPS):
            raise _not_allowed(node.op, node)
        _check_expr(node.operand)
        return
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check_expr(v)
        return
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _CMPOPS:
                raise _not_allowed(op, node)
        _check_expr(node.left)
        for c in node.comparators:
            _check_expr(c)
        return
    if isinstance(node, ast.IfExp):
        for n in (node.test, node.body, node.orelse):
            _check_expr(n)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _not_allowed(node.func)
        name = node.func.id
        if name not in FUNCTIONS:
            raise _err(node, f"function {name} is not allowed; allowed functions: "
                             f"{', '.join(sorted(FUNCTIONS))}")
        if node.keywords:
            raise _err(node, f"keyword arguments to {name} are not allowed")
        if not node.args:
            raise _err(node, f"{name} needs at least one argument")
        if name in ("relu", "sigmoid", "tanh", "abs") and len(node.args) != 1:
            raise _err(node, f"{name} takes exactly one argument")
        for a in node.args:
            if isinstance(a, (ast.Tuple, ast.List)):
                if name != "sum" or len(node.args) != 1:
                    raise _err(a, "a tuple/list is allowed only as the single argument of sum")
                for e in a.elts:
                    _check_expr(e)
            elif isinstance(a, ast.Starred):
                raise _not_allowed(a)
            else:
                _check_expr(a)
        return
    raise _not_allowed(node)


def _check_block(stmts: list[ast.stmt]) -> None:
    for i, st in enumerate(stmts):
        if isinstance(st, ast.Assign):
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                raise _err(st, "assignment is allowed only to a single plain name")
            _check_expr(st.value)
        elif isinstance(st, ast.If):
            _check_expr(st.test)
            _check_block(st.body)
            _check_block(st.orelse)
        elif isinstance(st, ast.Return):
            if st.value is None:
                raise _err(st, "return needs a value")
            _check_expr(st.value)
        elif isinstance(st, ast.Pass):
            pass
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            pass   # docstring / string comment
        else:
            raise _not_allowed(st)


def read_program(path: Path) -> str:
    """Program text from a file: UTF-8 (with or without BOM) or UTF-16 with BOM; anything else is a ProgramError."""
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ProgramError(f"{Path(path).name} is not UTF-8 text (bad byte at offset {e.start}); "
                           "save it as UTF-8") from None


@dataclass
class Program:
    src: str
    arg_names: list[str]
    body: list[ast.stmt]
    func: ast.FunctionDef


def parse_program(src: str, expected_args: list[str] | None = None) -> Program:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ProgramError(f"line {e.lineno}: syntax error: {e.msg}") from None
    funcs = []
    for st in tree.body:
        if isinstance(st, ast.FunctionDef):
            funcs.append(st)
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            continue
        else:
            raise _not_allowed(st)
    if len(funcs) != 1:
        raise _err(funcs[1] if len(funcs) > 1 else None,
                   "exactly one function is allowed: def net(x0, x1, ...)")
    f = funcs[0]
    if f.name != "net":
        raise _err(f, f"the function must be called net, not {f.name}")
    a = f.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.kw_defaults or a.posonlyargs:
        raise _err(f, "only plain positional arguments are allowed")
    if f.decorator_list:
        raise _err(f, "decorators are not allowed")
    args = [x.arg for x in a.args]
    if expected_args is not None and args != list(expected_args):
        raise _err(f, f"arguments must be ({', '.join(expected_args)}), got ({', '.join(args)})")
    _check_block(f.body)
    return Program(src, args, f.body, f)


class _Interp:
    def __init__(self, prog: Program, X: np.ndarray):
        self.X = X
        self.n = X.shape[0]
        self.env = {name: X[:, i].astype(np.float64) for i, name in enumerate(prog.arg_names)}
        self.assigned = {name: np.ones(self.n, bool) for name in prog.arg_names}
        self.out = np.full(self.n, np.nan)
        self.names = prog.arg_names

    @staticmethod
    def truth(v: np.ndarray) -> np.ndarray:
        return v if v.dtype == bool else (v != 0)

    @staticmethod
    def num(v: np.ndarray) -> np.ndarray:
        return v.astype(np.float64) if v.dtype == bool else v

    def block(self, stmts: list[ast.stmt], rows: np.ndarray) -> np.ndarray:
        for st in stmts:
            if rows.size == 0:
                break
            if isinstance(st, ast.Assign):
                name = st.targets[0].id
                v = self.num(self.expr(st.value, rows))
                if name not in self.env:
                    self.env[name] = np.full(self.n, np.nan)
                    self.assigned[name] = np.zeros(self.n, bool)
                self.env[name][rows] = v
                self.assigned[name][rows] = True
            elif isinstance(st, ast.Return):
                self.out[rows] = self.num(self.expr(st.value, rows))
                return rows[:0]
            elif isinstance(st, ast.If):
                c = self.truth(self.expr(st.test, rows))
                r1 = self.block(st.body, rows[c])
                r2 = self.block(st.orelse, rows[~c])
                rows = np.sort(np.concatenate([r1, r2]))
        return rows

    def expr(self, node: ast.AST, rows: np.ndarray) -> np.ndarray:
        k = rows.size
        if isinstance(node, ast.Constant):
            return np.full(k, float(node.value))
        if isinstance(node, ast.Name):
            if node.id not in self.env or not self.assigned[node.id][rows].all():
                raise _err(node, f"name {node.id} is used before it is assigned")
            return self.env[node.id][rows]
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self.num(self.expr(node.left, rows)),
                                          self.num(self.expr(node.right, rows)))
        if isinstance(node, ast.UnaryOp):
            v = self.expr(node.operand, rows)
            if isinstance(node.op, ast.Not):
                return ~self.truth(v)
            return -self.num(v) if isinstance(node.op, ast.USub) else self.num(v)
        if isinstance(node, ast.BoolOp):
            vals = [self.truth(self.expr(v, rows)) for v in node.values]
            f = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            return f.reduce(vals)
        if isinstance(node, ast.Compare):
            res = np.ones(k, bool)
            left = self.num(self.expr(node.left, rows))
            for op, comp in zip(node.ops, node.comparators):
                right = self.num(self.expr(comp, rows))
                res &= _CMPOPS[type(op)](left, right)
                left = right
            return res
        if isinstance(node, ast.IfExp):
            c = self.truth(self.expr(node.test, rows))
            return np.where(c, self.num(self.expr(node.body, rows)), self.num(self.expr(node.orelse, rows)))
        if isinstance(node, ast.Call):
            name = node.func.id
            if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
                args = [self.num(self.expr(e, rows)) for e in node.args[0].elts]
            else:
                args = [self.num(self.expr(a, rows)) for a in node.args]
            if name == "relu":
                return np.maximum(args[0], 0.0)
            if name == "sigmoid":
                return _sigmoid(args[0])
            if name == "tanh":
                return np.tanh(args[0])
            if name == "abs":
                return np.abs(args[0])
            if name == "min":
                return np.minimum.reduce(args)
            if name == "max":
                return np.maximum.reduce(args)
            if name == "sum":
                return np.add.reduce(args)
            if name == "argmax":
                return np.argmax(np.stack(args, axis=1), axis=1).astype(np.float64)
        raise _not_allowed(node)


def evaluate(prog: Program, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(prog.arg_names):
        raise ProgramError(f"expected inputs with {len(prog.arg_names)} columns, got shape {X.shape}")
    it = _Interp(prog, X)
    with np.errstate(all="ignore"):
        left = it.block(prog.body, np.arange(X.shape[0]))
    if left.size:
        r = left[0]
        example = ", ".join(f"{n}={X[r, i]:g}" for i, n in enumerate(prog.arg_names))
        raise ProgramError(f"no return for some inputs, e.g. {example}")
    return it.out


def program_size(prog: Program) -> dict:
    nodes = constants = relus = ifs = 0
    for st in prog.body:
        for n in ast.walk(st):
            nodes += 1
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                constants += 1
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "relu":
                relus += 1
            elif isinstance(n, (ast.If, ast.IfExp)):
                ifs += 1
    src_lines = prog.src.splitlines()[prog.func.lineno:prog.func.end_lineno]
    lines = sum(1 for ln in src_lines if ln.strip() and not ln.strip().startswith("#"))
    return {"lines": lines, "nodes": nodes, "constants": constants, "relus": relus, "ifs": ifs}
