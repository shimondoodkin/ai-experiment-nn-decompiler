"""Symbolic net: a dump program parsed back into units `act(affine)` and output affines.

A unit's act is relu, sigmoid, tanh or linear (no activation: an inlined or always-on unit, or an argmax
output score)."""
from __future__ import annotations

import ast
import copy as _copy
import math
from dataclasses import dataclass, field

import numpy as np

from .lang import parse_program
from .model import activate
from .spec import Spec


def fmt(c: float) -> str:
    c = float(c)
    if c == int(c) and abs(c) < 1e15:
        return str(int(c))
    return repr(c)


@dataclass
class Affine:
    coef: dict[str, float] = field(default_factory=dict)   # no zeros stored
    const: float = 0.0

    def eval(self, env: dict[str, np.ndarray]) -> np.ndarray:
        out = np.full(len(next(iter(env.values()))), self.const, dtype=np.float64)
        for v, c in self.coef.items():
            out = out + c * env[v]
        return out

    def interval(self, bounds: dict[str, tuple[float, float]]) -> tuple[float, float]:
        lo = hi = self.const
        for v, c in self.coef.items():
            a, b = bounds[v]
            lo += min(c * a, c * b)
            hi += max(c * a, c * b)
        return lo, hi

    def scaled(self, k: float) -> Affine:
        return Affine({v: c * k for v, c in self.coef.items() if c * k != 0}, self.const * k)

    def plus(self, other: Affine) -> Affine:
        coef = dict(self.coef)
        for v, c in other.coef.items():
            coef[v] = coef.get(v, 0.0) + c
        return Affine({v: c for v, c in coef.items() if c != 0}, self.const + other.const)

    def to_src(self) -> str:
        parts: list[str] = []
        for v, c in self.coef.items():
            mag = "" if abs(c) == 1 else f"{fmt(abs(c))} * "
            term = f"{mag}{v}"
            if not parts:
                parts.append(f"-{term}" if c < 0 else term)
            else:
                parts.append(f" {'-' if c < 0 else '+'} {term}")
        if self.const != 0 or not parts:
            k = self.const
            if not parts:
                parts.append(fmt(k))
            else:
                parts.append(f" {'-' if k < 0 else '+'} {fmt(abs(k))}")
        return "".join(parts)


@dataclass
class Unit:
    name: str
    layer: int
    pre: Affine
    act: str = "relu"      # relu, linear, sigmoid or tanh


UNIT_ACTS = ("relu", "sigmoid", "tanh")   # the calls from_dump reads around an affine expression


def _sigmoid(z: float) -> float:
    e = math.exp(-abs(z))          # stable: never overflows
    return 1 / (1 + e) if z >= 0 else e / (1 + e)


def post_interval(act: str, lo: float, hi: float) -> tuple[float, float]:
    """Interval of act(z) for z in [lo, hi] (every act is monotone non-decreasing)."""
    if act == "relu":
        return (max(lo, 0.0), max(hi, 0.0))
    if act == "linear":
        return (lo, hi)
    if act == "sigmoid":
        return (_sigmoid(lo), _sigmoid(hi))
    if act == "tanh":
        return (math.tanh(lo), math.tanh(hi))
    raise ValueError(f"unknown activation {act!r}")


@dataclass
class SymNet:
    inputs: list[str]
    units: list[Unit]
    outputs: list[Affine]
    kind: str
    threshold: float = 0.0

    def copy(self) -> SymNet:
        return _copy.deepcopy(self)

    def unit(self, name: str) -> Unit:
        for u in self.units:
            if u.name == name:
                return u
        raise KeyError(name)


def _affine(node: ast.AST) -> Affine:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Affine({}, float(node.value))
    if isinstance(node, ast.Name):
        return Affine({node.id: 1.0})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        a = _affine(node.operand)
        return a.scaled(-1.0) if isinstance(node.op, ast.USub) else a
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        r = _affine(node.right)
        return _affine(node.left).plus(r.scaled(-1.0) if isinstance(node.op, ast.Sub) else r)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a, b = _affine(node.left), _affine(node.right)
        if not a.coef:
            return b.scaled(a.const)
        if not b.coef:
            return a.scaled(b.const)
    raise ValueError(f"line {getattr(node, 'lineno', '?')}: not an affine expression")


def from_dump(src: str, spec: Spec) -> SymNet:
    prog = parse_program(src, spec.arg_names)
    layer_of = {name: 0 for name in spec.arg_names}
    units: list[Unit] = []
    outputs: list[Affine] | None = None
    for st in prog.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue
        if isinstance(st, ast.Assign):
            name = st.targets[0].id
            v = st.value
            is_act = isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in UNIT_ACTS
            act = v.func.id if is_act else "linear"
            pre = _affine(v.args[0] if is_act else v)
            unknown = [x for x in pre.coef if x not in layer_of]
            if unknown:
                raise ValueError(f"line {st.lineno}: unknown name {unknown[0]}")
            layer = 1 + max((layer_of[x] for x in pre.coef), default=0)
            layer_of[name] = layer
            units.append(Unit(name, layer, pre, act))
        elif isinstance(st, ast.Return) and outputs is None:
            v = st.value
            is_argmax = isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "argmax"
            if is_argmax != (spec.output == "argmax"):
                raise ValueError(f"line {st.lineno}: return shape does not match output kind {spec.output}")
            if is_argmax:
                names = [a.id for a in v.args if isinstance(a, ast.Name)]
                outs = {u.name: u for u in units}
                if len(names) != len(v.args) or any(n not in outs or outs[n].act != "linear" for n in names):
                    raise ValueError(f"line {st.lineno}: argmax must be over linear output units")
                outputs = [outs[n].pre for n in names]
                units = [u for u in units if u.name not in names]
            else:
                outputs = [_affine(v)]
        else:
            raise ValueError(f"line {getattr(st, 'lineno', '?')}: not a dump-shaped statement")
    if outputs is None:
        raise ValueError("line ?: dump has no return statement")
    return SymNet(list(spec.arg_names), units, outputs, spec.output, spec.threshold)


def evaluate(net: SymNet, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    env = {n: X[:, i] for i, n in enumerate(net.inputs)}
    for u in net.units:
        z = u.pre.eval(env)
        env[u.name] = z if u.act == "linear" else activate(u.act, z)
    return np.stack([o.eval(env) for o in net.outputs], axis=1)


def input_bounds(spec: Spec) -> dict[str, tuple[float, float]]:
    return {i.name: (float(i.lo), float(i.hi)) for i in spec.inputs}


def unit_bounds(net: SymNet, spec: Spec) -> dict[str, tuple[float, float]]:
    """Pre-activation interval of each unit over the input box."""
    post = input_bounds(spec)
    pre: dict[str, tuple[float, float]] = {}
    for u in net.units:
        lo, hi = u.pre.interval(post)
        pre[u.name] = (lo, hi)
        post[u.name] = post_interval(u.act, lo, hi)
    return pre


def substitute(net: SymNet, name: str, value: Affine) -> SymNet:
    """Remove unit `name` and replace its output everywhere by `value`."""
    out = net.copy()

    def sub(a: Affine) -> Affine:
        c = a.coef.get(name)
        if c is None:
            return a
        rest = Affine({v: k for v, k in a.coef.items() if v != name}, a.const)
        return rest.plus(value.scaled(c))

    out.units = [Unit(u.name, u.layer, sub(u.pre), u.act) for u in out.units if u.name != name]
    out.outputs = [sub(o) for o in out.outputs]
    return out


def to_program(net: SymNet) -> str:
    lines = [f"def net({', '.join(net.inputs)}):"]
    for u in net.units:
        body = u.pre.to_src() if u.act == "linear" else f"{u.act}({u.pre.to_src()})"
        lines.append(f"    {u.name} = {body}")
    if net.kind == "argmax":
        names = []
        for k, o in enumerate(net.outputs):
            lines.append(f"    o{k} = {o.to_src()}")
            names.append(f"o{k}")
        lines.append(f"    return argmax({', '.join(names)})")
    else:
        lines.append(f"    return {net.outputs[0].to_src()}")
    return "\n".join(lines) + "\n"
