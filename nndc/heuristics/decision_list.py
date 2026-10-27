"""P10: argmax outputs as a decision list; each test is a binary problem on one score difference.

Only the ordering of the scores matters, so "class a wins" is o_a - o_b > 0 against the class b it
borders most. Classes that never touch need no comparison, and argmax becomes an if-chain. Each
test is solved like a binary output: boundary (P6), boundary scan (P6b), truth table (P7, bool
domains), or per linear region: fold o_a - o_b through the activation pattern most common on that
boundary, which is exact when the boundary sits inside one pattern.
HEURISTICS: iris_petal/seed0, iris/seed0, xor/seed0, window/seed1. Approximate.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import replace

import numpy as np

from ..check import build_input_set, net_output
from ..lang import evaluate, parse_program
from ..model import MLP
from ..spec import DISCRETE_TYPES, Spec
from ..symnet import Affine, SymNet, fmt, input_bounds
from ..symnet import evaluate as net_eval
from .boundary import boundary_expr
from .boundary_scan import scan_boundary_expr
from .cleanup import join_terms, var_items
from .snap import nice
from .truthtable import _library

MAX_CLASSES = 4            # permutations of the class order: 4! = 24
NEAR_FRAC = 0.05           # share of a step's points closest to o_a = o_b that vote for the pattern
NEAR_MIN = 20
TOP_PATTERNS = 3
MIN_AGREE = 0.9            # below this the whole route is not applicable
SNAP_LADDER = [(0.05, 0.01), (0.02, 0.005), (0.01, 0.0025)]   # (coefficient, threshold) relative tolerances


def fold(net: SymNet, on: dict[str, bool], a: Affine) -> Affine:
    """Affine form of `a` over the inputs inside one activation pattern (relu = identity or 0)."""
    for u in reversed(net.units):
        c = a.coef.get(u.name)
        if c is None:
            continue
        rest = Affine({v: k for v, k in a.coef.items() if v != u.name}, a.const)
        a = rest.plus(u.pre.scaled(c)) if (not u.relu or on[u.name]) else rest
    return a


def _patterns(net: SymNet, X: np.ndarray) -> tuple[list[str], np.ndarray]:
    env = {n: X[:, i] for i, n in enumerate(net.inputs)}
    names, cols = [], []
    for u in net.units:
        z = u.pre.eval(env)
        env[u.name] = np.maximum(z, 0.0) if u.relu else z
        if u.relu:
            names.append(u.name)
            cols.append(z > 0)
    return names, (np.stack(cols, axis=1) if cols else np.zeros((len(X), 0), bool))


def _test_src(a: Affine, spec: Spec, snap: tuple[float, float] | None = None) -> str:
    """`a > 0` written as `lhs > rhs` with the first input's coefficient 1 (and optionally snapped)."""
    first = next((v for v in spec.arg_names if v in a.coef), None)
    if first is None:
        return "1" if a.const > 0 else "0"
    c0 = a.coef[first]
    u = a.scaled(1.0 / abs(c0))
    op = ">" if c0 > 0 else "<"
    if c0 < 0:
        u, op = u.scaled(-1.0), "<"
    coef, t = dict(u.coef), -u.const
    if snap is not None:
        crel, trel = snap
        coef = {v: (c if v == first else nice(c, crel * abs(c))) for v, c in coef.items()}
        coef = {v: c for v, c in coef.items() if c != 0}
        lo, hi = Affine(coef).interval(input_bounds(spec))
        t = nice(t, trel * max(hi - lo, 1e-12))
    lhs = join_terms(var_items(Affine({v: coef[v] for v in spec.arg_names if v in coef})))
    return f"{lhs} {op} {fmt(t)}"


def _program(steps: list[tuple[str, dict[str, str], int]], last: int, spec: Spec) -> str:
    lines = [f"def net({', '.join(spec.arg_names)}):"]
    for k, (_, derived, _) in enumerate(steps):
        for name, value in derived.items():
            lines.append(f"    {re.sub(r'\bd0\b', f'd{k}', name)} = {value}")
    for k, (test, _, cls) in enumerate(steps):
        lines.append(f"    if {re.sub(r'\bd0\b', f'd{k}', test)}:")
        lines.append(f"        return {cls}")
    lines.append(f"    return {last}")
    return "\n".join(lines) + "\n"


def _holds(expr: str, derived: dict[str, str], spec: Spec, X: np.ndarray) -> np.ndarray:
    body = [f"    {n} = {v}" for n, v in derived.items()]
    src = "\n".join([f"def net({', '.join(spec.arg_names)}):", *body, f"    return {expr}"]) + "\n"
    return evaluate(parse_program(src, spec.arg_names), X) > 0.5


def decision_list(net: SymNet, spec: Spec, mlp: MLP) -> tuple[str, list[str], list[str]] | None:
    """(program with exact folds, snapped programs coarsest first, trace) or None when not applicable.

    `net` is the pruned symbolic net; `mlp` is the original, used only for the checker's input set
    (uniform points, corners and points on unit and class boundaries) and its decisions.
    """
    if spec.output != "argmax":
        return None
    X, _ = build_input_set(mlp, spec, seed=0)
    y = net_output(mlp, spec, X)[0].astype(int)
    out = net_eval(net, X)
    classes = [int(c) for c in np.unique(y)]
    if len(classes) == 1:
        src = f"def net({', '.join(spec.arg_names)}):\n    return {classes[0]}\n"
        return src, [], [f"decision list: one class ({classes[0]}) on the whole box"]
    if len(classes) > MAX_CLASSES:
        return None
    names, P = _patterns(net, X)
    bool_domain = all(i.type == "bool" for i in spec.inputs)
    continuous = not any(i.type in DISCRETE_TYPES for i in spec.inputs)
    bspec = replace(spec, output="binary", n_outputs=1, threshold=0.0)
    cache: dict[tuple[int, frozenset], tuple] = {}

    def step(a: int, rest: frozenset) -> tuple | None:
        """Best test for 'class a' among points labelled in rest: (expr, derived, agreement, fold affine)."""
        key = (a, rest)
        if key in cache:
            return cache[key]
        R = np.isin(y, list(rest))
        want = y[R] == a
        others = [c for c in rest if c != a]
        sub = out[R][want][:, others]
        b = others[int(Counter(np.argmax(sub, axis=1)).most_common(1)[0][0])] if len(sub) else others[0]
        s_aff = net.outputs[a].plus(net.outputs[b].scaled(-1.0))
        s = out[R][:, a] - out[R][:, b]
        cands: list[tuple[str, dict[str, str], Affine | None, str]] = []
        n_near = max(NEAR_MIN, int(NEAR_FRAC * R.sum()))
        near = np.argsort(np.abs(s))[:n_near]
        for pat, _ in Counter(map(tuple, P[R][near])).most_common(TOP_PATTERNS):
            aff = fold(net, dict(zip(names, pat)), s_aff)
            cands.append((_test_src(aff, spec), {}, aff, "fold"))
        bnet = SymNet(net.inputs, net.units, [s_aff], "binary", 0.0)
        try:
            r = boundary_expr(bnet, bspec)
        except ValueError:
            r = None
        if r is not None:
            cands.append((r[0], r[1], None, "boundary"))
        if continuous and cands:
            dirs = [dict(c[2].coef) for c in cands if c[2] is not None and c[2].coef]
            dirs += [dict(u.pre.coef) for u in net.units if set(u.pre.coef) <= set(net.inputs) and u.pre.coef]
            dirs += [{v: 1.0} for v in net.inputs]
            r = scan_boundary_expr(lambda Z: net_eval(bnet, Z)[:, 0], bspec, dirs)
            if r is not None:
                cands.append((r[0], r[1], None, "scan"))
        if bool_domain:
            for expr in _library(spec.arg_names):
                if np.array_equal(_holds(expr, {}, spec, X[R]), want):
                    cands.append((expr, {}, None, "truth-table"))
                    break
        scored = []
        for expr, derived, aff, how in cands:
            if expr is None:
                continue
            agree = float((_holds(expr, derived, spec, X[R]) == want).mean())
            scored.append((-agree, len(expr) + sum(len(v) for v in derived.values()), expr, derived, aff, how, b))
        best = min(scored, key=lambda t: t[:2]) if scored else None
        cache[key] = best
        return best

    trace: list[str] = []
    best_order = None
    for order in itertools.permutations(classes):
        chosen = []
        for k, a in enumerate(order[:-1]):
            r = step(a, frozenset(order[k:]))
            if r is None:
                break
            chosen.append(r)
        else:
            src = _program([(c[2], c[3], a) for c, a in zip(chosen, order)], order[-1], spec)
            agree = float((evaluate(parse_program(src, spec.arg_names), X) == y).mean())
            key = (-agree, len(src))
            if best_order is None or key < best_order[0]:
                best_order = (key, order, chosen, src)
    if best_order is None or -best_order[0][0] < MIN_AGREE:
        return None
    (neg, _), order, chosen, src = best_order
    trace.append(f"decision list: order {list(order)}, agreement with the net on the sample {-neg:.4f}")
    for c, a in zip(chosen, order):
        trace.append(f"decision list: class {a} vs class {c[6]} by {c[5]}: {c[2]} (agreement {-c[0]:.4f})")
    snapped = []
    for tol in SNAP_LADDER:
        steps = [((_test_src(c[4], spec, tol) if c[4] is not None else c[2]), c[3], a) for c, a in zip(chosen, order)]
        s = _program(steps, order[-1], spec)
        if s != src and s not in snapped:
            snapped.append(s)
    return src, snapped, trace

