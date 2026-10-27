"""Compile a program back into a ReLU MLP: the "reinsert" step of the fix experiment.

1. The function body is flattened into one expression: assignments are substituted, and an `if`
   becomes a conditional expression whose branches each continue with the rest of the body.
2. Values compile to affine combinations of inputs and ReLU nodes: relu, max(a, b) = b + relu(a - b),
   min(a, b) = a - relu(a - b), abs(e) = relu(e) + relu(-e). Comparisons used as numbers, and
   conditionals on values, need a step: a ramp of width RAMP_REL of the margin's range (the only
   non-exact construct) and a big-M gate for a non-constant branch difference.
3. Conditions compile to margins that are positive exactly when true: a < b -> b - a, and -> min,
   or -> max, not -> negation; == / != only between integer-valued expressions. Binary programs that
   return a condition output threshold + margin; argmax decision lists output, per class, the max over
   the paths reaching it of the min of the path's margins. These are exact except on exact ties.
4. ReLU nodes are layered by depth; values a deeper layer still needs are carried through with
   identity units (relu(v) for a ReLU node, the pair relu(x), relu(-x) for an input).
"""
from __future__ import annotations

import ast

import numpy as np

from .lang import parse_program
from .model import MLP
from .spec import DISCRETE_TYPES, Spec

RAMP_REL = 1e-6


class CompileError(ValueError):
    pass


# ---- 1. flatten statements into one expression -------------------------------------------------

def _subst(node: ast.AST, env: dict[str, ast.AST]) -> ast.AST:
    if isinstance(node, ast.Name):
        return env.get(node.id, node)
    if isinstance(node, ast.Constant):
        return node
    if isinstance(node, ast.BinOp):
        return ast.BinOp(_subst(node.left, env), node.op, _subst(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(node.op, _subst(node.operand, env))
    if isinstance(node, ast.BoolOp):
        return ast.BoolOp(node.op, [_subst(v, env) for v in node.values])
    if isinstance(node, ast.Compare):
        return ast.Compare(_subst(node.left, env), node.ops, [_subst(c, env) for c in node.comparators])
    if isinstance(node, ast.IfExp):
        return ast.IfExp(_subst(node.test, env), _subst(node.body, env), _subst(node.orelse, env))
    if isinstance(node, ast.Call):
        args = node.args[0].elts if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)) \
            else node.args
        return ast.Call(node.func, [_subst(a, env) for a in args], [])
    raise CompileError(f"unsupported expression {ast.unparse(node)}")


def _flatten(stmts: list[ast.stmt], env: dict[str, ast.AST]) -> ast.AST:
    for i, st in enumerate(stmts):
        if isinstance(st, ast.Assign):
            env = env | {st.targets[0].id: _subst(st.value, env)}
        elif isinstance(st, ast.Return):
            return _subst(st.value, env)
        elif isinstance(st, ast.If):
            rest = stmts[i + 1:]
            return ast.IfExp(_subst(st.test, env), _flatten(st.body + rest, env), _flatten(st.orelse + rest, env))
        elif isinstance(st, ast.Expr):   # a docstring
            continue
        else:
            raise CompileError(f"unsupported statement {ast.unparse(st)}")
    raise CompileError("a path through the program has no return")


# ---- 2/3. affine forms over inputs and ReLU nodes ------------------------------------------------

class _Builder:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.d = len(spec.inputs)
        self.relus: list[tuple[dict, float]] = []   # ReLU node k: its argument (coefs over atoms, const)
        self.keys: dict = {}
        self.ivals: dict = {}
        self.ramps = 0
        self.discrete = [i.type in DISCRETE_TYPES for i in spec.inputs]

    # a linear form is (coefs: {atom: c}, const); atoms are ("x", i) and ("r", k)
    @staticmethod
    def const(c: float) -> tuple[dict, float]:
        return {}, float(c)

    @staticmethod
    def add(a, b, s: float = 1.0):
        coefs = dict(a[0])
        for k, v in b[0].items():
            coefs[k] = coefs.get(k, 0.0) + s * v
        return {k: v for k, v in coefs.items() if v != 0}, a[1] + s * b[1]

    @staticmethod
    def scale(a, s: float):
        return ({k: s * v for k, v in a[0].items()} if s != 0 else {}), a[1] * s

    def relu(self, a):
        if not a[0]:
            return self.const(max(a[1], 0.0))
        lo, hi = self.interval(a)
        if lo >= 0:
            return a
        if hi <= 0:
            return self.const(0.0)
        key = (tuple(sorted(a[0].items())), a[1])
        if key not in self.keys:
            self.keys[key] = len(self.relus)
            self.relus.append(a)
        return {("r", self.keys[key]): 1.0}, 0.0

    def interval(self, a) -> tuple[float, float]:
        lo = hi = a[1]
        for atom, c in a[0].items():
            alo, ahi = self.atom_interval(atom)
            lo += min(c * alo, c * ahi)
            hi += max(c * alo, c * ahi)
        return lo, hi

    def atom_interval(self, atom) -> tuple[float, float]:
        if atom[0] == "x":
            i = self.spec.inputs[atom[1]]
            return float(i.lo), float(i.hi)
        if atom not in self.ivals:
            lo, hi = self.interval(self.relus[atom[1]])
            self.ivals[atom] = (max(lo, 0.0), max(hi, 0.0))
        return self.ivals[atom]

    def vmax(self, a, b):
        return self.add(b, self.relu(self.add(a, b, -1)))

    def vmin(self, a, b):
        return self.add(a, self.relu(self.add(a, b, -1)), -1)

    def integer_valued(self, a) -> bool:
        return (all(k[0] == "x" and self.discrete[k[1]] and float(v).is_integer() for k, v in a[0].items())
                and float(a[1]).is_integer())

    # values
    def val(self, e: ast.AST):
        if isinstance(e, ast.Constant):
            return self.const(float(e.value))
        if isinstance(e, ast.Name):
            return {("x", self.spec.arg_names.index(e.id)): 1.0}, 0.0
        if isinstance(e, ast.BinOp):
            a, b = self.val(e.left), self.val(e.right)
            if isinstance(e.op, ast.Add):
                return self.add(a, b)
            if isinstance(e.op, ast.Sub):
                return self.add(a, b, -1)
            if isinstance(e.op, ast.Mult):
                if not a[0]:
                    return self.scale(b, a[1])
                if not b[0]:
                    return self.scale(a, b[1])
                raise CompileError(f"product of two non-constant terms: {ast.unparse(e)}")
            if isinstance(e.op, ast.Div) and not b[0]:
                return self.scale(a, 1.0 / b[1])
            raise CompileError(f"unsupported operation {ast.unparse(e)}")
        if isinstance(e, ast.UnaryOp):
            if isinstance(e.op, ast.Not):
                return self.step(self.margin(e))
            v = self.val(e.operand)
            return self.scale(v, -1.0) if isinstance(e.op, ast.USub) else v
        if isinstance(e, ast.Call):
            name, args = e.func.id, [self.val(a) for a in e.args]
            if name == "relu":
                return self.relu(args[0])
            if name == "abs":
                return self.add(self.relu(args[0]), self.relu(self.scale(args[0], -1.0)))
            if name in ("max", "min"):
                out = args[0]
                for b in args[1:]:
                    out = self.vmax(out, b) if name == "max" else self.vmin(out, b)
                return out
            if name == "sum":
                out = self.const(0.0)
                for b in args:
                    out = self.add(out, b)
                return out
            raise CompileError(f"{name}() is only supported as the returned value of an argmax program")
        if isinstance(e, (ast.Compare, ast.BoolOp)):
            return self.step(self.margin(e))
        if isinstance(e, ast.IfExp):
            s = self.step(self.margin(e.test))
            a, b = self.val(e.body), self.val(e.orelse)
            return self.add(b, self.gate(s, self.add(a, b, -1)))
        raise CompileError(f"unsupported expression {ast.unparse(e)}")

    def step(self, m):
        """1 where m > 0, 0 where m <= 0, with a ramp of width RAMP_REL of m's range in between."""
        lo, hi = self.interval(m)
        if lo > 0:
            return self.const(1.0)
        if hi <= 0:
            return self.const(0.0)
        self.ramps += 1
        z = self.scale(m, 1.0 / (RAMP_REL * (hi - lo)))
        # 1 - relu(1 - relu(z)) equals relu(z) - relu(z - 1), but is exactly 0 or 1 outside the ramp
        # in float32: the steep z never meets a large value it must cancel against
        return self.add(self.const(1.0), self.relu(self.add(self.const(1.0), self.relu(z), -1)), -1)

    def gate(self, s, d):
        """s * d for s in {0, 1}: exact at the ends, a big-M construction for a non-constant d."""
        if not d[0]:
            return self.scale(s, d[1])
        lo, hi = self.interval(d)
        big = max(abs(lo), abs(hi))
        slack = self.scale(self.add(self.const(1.0), s, -1), big)          # M (1 - s)
        return self.add(self.relu(self.add(d, slack, -1)), self.relu(self.add(self.scale(d, -1.0), slack, -1)), -1)

    # conditions
    def margin(self, e: ast.AST):
        if isinstance(e, ast.Compare):
            parts, left = [], self.val(e.left)
            for op, comp in zip(e.ops, e.comparators):
                right = self.val(comp)
                if isinstance(op, (ast.Lt, ast.LtE)):
                    parts.append(self.add(right, left, -1))
                elif isinstance(op, (ast.Gt, ast.GtE)):
                    parts.append(self.add(left, right, -1))
                else:
                    diff = self.add(left, right, -1)
                    if not self.integer_valued(diff):
                        raise CompileError(f"== / != between non-integer values: {ast.unparse(e)}")
                    a = self.add(self.relu(diff), self.relu(self.scale(diff, -1.0)))
                    parts.append(self.add(self.const(0.5), a, -1) if isinstance(op, ast.Eq)
                                 else self.add(a, self.const(0.5), -1))
                left = right
            return self._fold(parts, self.vmin)
        if isinstance(e, ast.BoolOp):
            ms = [self.margin(v) for v in e.values]
            return self._fold(ms, self.vmin if isinstance(e.op, ast.And) else self.vmax)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            return self.scale(self.margin(e.operand), -1.0)
        if isinstance(e, ast.Constant):
            return self.const(1.0 if e.value else -1.0)
        if isinstance(e, ast.IfExp):
            c, a, b = self.margin(e.test), self.margin(e.body), self.margin(e.orelse)
            return self.vmax(self.vmin(c, a), self.vmin(self.scale(c, -1.0), b))
        v = self.val(e)   # a number used as a truth value: true when nonzero
        if not self.integer_valued(v):
            raise CompileError(f"a non-integer number used as a condition: {ast.unparse(e)}")
        return self.add(self.add(self.relu(v), self.relu(self.scale(v, -1.0))), self.const(0.5), -1)

    @staticmethod
    def _fold(items, f):
        out = items[0]
        for b in items[1:]:
            out = f(out, b)
        return out


def _is_condition(e: ast.AST, threshold: float) -> bool:
    if isinstance(e, (ast.Compare, ast.BoolOp)) or (isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not)):
        return True
    if isinstance(e, ast.Constant):
        return isinstance(e.value, bool) or (e.value in (0, 1) and 0 <= threshold < 1)
    if isinstance(e, ast.IfExp):
        return _is_condition(e.body, threshold) and _is_condition(e.orelse, threshold)
    return False


def _class_paths(e: ast.AST, path: list, b: _Builder, out: dict[int, list]) -> None:
    if isinstance(e, ast.IfExp):
        m = b.margin(e.test)
        _class_paths(e.body, path + [m], b, out)
        _class_paths(e.orelse, path + [b.scale(m, -1.0)], b, out)
        return
    if not (isinstance(e, ast.Constant) and float(e.value).is_integer()):
        raise CompileError(f"an argmax decision list must return class indices, got {ast.unparse(e)}")
    out.setdefault(int(e.value), []).append(b._fold(path, b.vmin) if path else b.const(1.0))


# ---- 4. layering ---------------------------------------------------------------------------------

def _layer(b: _Builder, outputs: list) -> tuple[MLP, list[int]]:
    depth: dict = {}

    def dep(atom) -> int:
        if atom[0] == "x":
            return 0
        if atom not in depth:
            depth[atom] = 1 + max((dep(a) for a in b.relus[atom[1]][0]), default=0)
        return depth[atom]

    atoms = [("r", k) for k in range(len(b.relus))]
    D = max((dep(a) for a in atoms), default=0)
    needed_after = [set() for _ in range(D + 1)]   # atoms used by layer > l or by the outputs
    for a in atoms:
        for used in b.relus[a[1]][0]:
            for l in range(dep(used), dep(a) - 1):
                needed_after[l + 1].add(used)
    for f in outputs:
        for used in f[0]:
            for l in range(dep(used) + 1, D + 1):
                needed_after[l].add(used)
    rep = {("x", i): {i: 1.0} for i in range(b.d)}
    n_prev, Ws, bs, widths = b.d, [], [], []

    def row(form) -> tuple[np.ndarray, float]:
        w = np.zeros(n_prev)
        for atom, c in form[0].items():
            for u, wu in rep[atom].items():
                w[u] += c * wu
        return w, form[1]

    for l in range(1, D + 1):
        rows, new_rep = [], {}
        for a in atoms:
            if dep(a) == l:
                new_rep[a] = {len(rows): 1.0}
                rows.append(row(b.relus[a[1]]))
        for a in sorted(needed_after[l], key=str):
            if dep(a) >= l:
                continue
            if a[0] == "r":
                new_rep[a] = {len(rows): 1.0}
                rows.append(row(({a: 1.0}, 0.0)))
            else:
                new_rep[a] = {len(rows): 1.0, len(rows) + 1: -1.0}
                rows.append(row(({a: 1.0}, 0.0)))
                rows.append(row(({a: -1.0}, 0.0)))
        Ws.append(np.array([r[0] for r in rows]).reshape(len(rows), n_prev))
        bs.append(np.array([r[1] for r in rows]))
        rep, n_prev = new_rep, len(rows)
        widths.append(len(rows))
    outs = [row(f) for f in outputs]
    Ws.append(np.array([r[0] for r in outs]).reshape(len(outs), n_prev))
    bs.append(np.array([r[1] for r in outs]))
    return MLP([W.astype(np.float32) for W in Ws], [v.astype(np.float32) for v in bs]), widths


def compile_program(src: str, spec: Spec) -> tuple[MLP, dict]:
    prog = parse_program(src, spec.arg_names)
    expr = _flatten(prog.body, {})
    b = _Builder(spec)
    if spec.output == "argmax":
        if isinstance(expr, ast.Call) and expr.func.id == "argmax":
            outputs = [b.val(a) for a in expr.args]
        else:
            paths: dict[int, list] = {}
            _class_paths(expr, [], b, paths)
            scores = {k: b._fold(ms, b.vmax) for k, ms in paths.items()}
            if any(k < 0 or k >= spec.n_outputs for k in scores):
                raise CompileError(f"class index outside 0..{spec.n_outputs - 1}")
            big = 1.0 + max(max(abs(v) for v in b.interval(s)) for s in scores.values())
            outputs = [scores.get(k, b.const(-big)) for k in range(spec.n_outputs)]
    elif spec.output == "binary" and _is_condition(expr, spec.threshold):
        outputs = [b.add(b.const(spec.threshold), b.margin(expr))]
    else:
        outputs = [b.val(expr)]
    mlp, widths = _layer(b, outputs)
    return mlp, {"exact": b.ramps == 0, "ramps": b.ramps, "hidden": widths}
