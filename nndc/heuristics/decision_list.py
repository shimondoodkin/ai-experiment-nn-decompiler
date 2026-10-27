"""P10: argmax outputs as a decision list; each test is a binary problem on one score difference.

Only the ordering of the scores matters, so "class a wins" is o_a - o_b > 0 against the class b it
borders most. Classes that never touch need no comparison, and argmax becomes an if-chain. Each
test is solved like a binary output: boundary (P6), boundary scan (P6b), truth table (P7, bool
domains), or per linear region: fold o_a - o_b through the activation pattern most common on that
boundary, which is exact when the boundary sits inside one pattern.

v2 (after the iris held-out run): a boundary with a kink is two half-planes joined by `or`/`and`,
one per activation pattern; and a threshold is placed in the middle of the range the net's confident
points allow (snapped: the nicest value inside that range), measured on a sample independent of
the checker's.
HEURISTICS: iris_petal/seed0-2, iris/seed0-2, xor/seed0, window/seed1. Approximate.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from ..check import build_input_set, net_output
from ..lang import evaluate, parse_program, program_size
from ..model import MLP
from ..spec import DISCRETE_TYPES, Spec
from ..symnet import Affine, SymNet, fmt, input_bounds
from ..symnet import evaluate as net_eval
from .boundary import boundary_expr
from .boundary_scan import scan_boundary_expr
from .cleanup import join_terms, var_items
from .snap import NICE_DENOMS, nice

MAX_CLASSES = 4            # permutations of the class order: 4! = 24
NEAR_FRAC = 0.05           # share of a step's points closest to o_a = o_b that vote for the pattern
NEAR_MIN = 20
TOP_PATTERNS = 4
PAIR_FROM = 3              # two-part tests combine the first 3 distinct fold half-planes
MIN_AGREE = 0.9            # below this the whole route is not applicable
SAMPLE_SEED = 1            # the tool's own sample; the checker uses seed 0
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


@dataclass
class Half:
    """Half-plane `w.x op t` with the first input's coefficient exactly 1."""
    w: dict[str, float]
    op: str
    t: float

    def g(self, X: np.ndarray, spec: Spec) -> np.ndarray:
        return sum(c * X[:, spec.arg_names.index(v)] for v, c in self.w.items())

    def holds(self, X: np.ndarray, spec: Spec, t: float | None = None) -> np.ndarray:
        g, t = self.g(X, spec), self.t if t is None else t
        return g > t if self.op == ">" else g < t

    def src(self, spec: Spec, t: float | None = None) -> str:
        lhs = join_terms(var_items(Affine({v: self.w[v] for v in spec.arg_names if v in self.w})))
        return f"{lhs} {self.op} {fmt(self.t if t is None else t)}"

    def snapped(self, crel: float) -> Half:
        first = next(iter(self.w))
        w = {v: (c if v == first else nice(c, crel * abs(c))) for v, c in self.w.items()}
        return Half({v: c for v, c in w.items() if c != 0}, self.op, self.t)

    def flipped(self) -> Half:
        return Half(self.w, "<" if self.op == ">" else ">", self.t)


def half(a: Affine, spec: Spec) -> Half | None:
    """`a > 0` as a half-plane, or None when `a` is constant."""
    first = next((v for v in spec.arg_names if v in a.coef), None)
    if first is None:
        return None
    c0 = a.coef[first]
    w = {first: 1.0} | {v: c / abs(c0) for v in spec.arg_names if (c := a.coef.get(v)) and v != first}
    t = -a.const / abs(c0)
    if c0 < 0:
        return Half({v: (c if v == first else -c) for v, c in w.items()}, "<", -t)
    return Half(w, ">", t)


def nicest_in(lo: float, hi: float) -> float:
    """The simplest number strictly inside (lo, hi): 0, else the coarsest k/d, nearest the middle."""
    if lo < 0 < hi:
        return 0.0
    mid = (lo + hi) / 2
    for d in NICE_DENOMS:
        k0, k1 = math.floor(lo * d) + 1, math.ceil(hi * d) - 1
        if k0 <= k1:
            return min(max(round(mid * d), k0), k1) / d
    return mid


def threshold_ranges(halves: list[Half], join: str, X: np.ndarray, want: np.ndarray,
                     spec: Spec) -> list[tuple[float, float]] | None:
    """Per half, the thresholds that classify every given point right; None if there are none.

    `or`: points of the other classes must fail every half, and each point of this class needs one
    half (the one it is deeper inside). `and` is the `or` of the flipped halves for the other classes.
    """
    if join == "and":
        return threshold_ranges([h.flipped() for h in halves], "or", X, ~want, spec)
    box = input_bounds(spec)
    G = [h.g(X, spec) for h in halves]
    neg_bound = []
    for h, g in zip(halves, G):
        lo, hi = Affine(h.w).interval(box)
        gn = g[~want]
        if h.op == ">":
            neg_bound.append(gn.max() if gn.size else lo)
        else:
            neg_bound.append(gn.min() if gn.size else hi)
    slack = np.stack([(g - nb) if h.op == ">" else (nb - g) for h, g, nb in zip(halves, G, neg_bound)], axis=1)
    owner = np.argmax(slack, axis=1)
    out = []
    for i, (h, g, nb) in enumerate(zip(halves, G, neg_bound)):
        lo, hi = Affine(h.w).interval(box)
        gp = g[want & (owner == i)]
        if h.op == ">":
            r = (nb, gp.min() if gp.size else hi)
        else:
            r = (gp.max() if gp.size else lo, nb)
        if not r[0] < r[1]:
            return None
        out.append((float(r[0]), float(r[1])))
    return out


@dataclass
class Test:
    how: str
    halves: list[Half] = field(default_factory=list)   # fold tests: halves joined by `join`
    join: str = "or"
    expr: str = ""                                       # other solvers: the written test
    derived: dict[str, str] = field(default_factory=dict)

    def src(self, spec: Spec, ts: list[float] | None = None) -> str:
        if not self.halves:
            return self.expr
        ts = ts or [h.t for h in self.halves]
        return f" {self.join} ".join(h.src(spec, t) for h, t in zip(self.halves, ts))

    def holds(self, X: np.ndarray, spec: Spec, ts: list[float] | None = None) -> np.ndarray:
        if not self.halves:
            return _holds(self.expr, self.derived, spec, X)
        ts = ts or [h.t for h in self.halves]
        parts = [h.holds(X, spec, t) for h, t in zip(self.halves, ts)]
        return np.logical_or.reduce(parts) if self.join == "or" else np.logical_and.reduce(parts)


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


@dataclass
class Step:
    test: Test
    other: int                 # the neighbouring class b of o_a - o_b
    agree: float               # on the confident points of this step, thresholds centred
    X: np.ndarray              # confident points of this step
    want: np.ndarray

    def written(self, spec: Spec, mode: str, snap: tuple[float, float] | None = None) -> str:
        """mode: exact (fold constants), centred (middle of the allowed range) or snapped."""
        t = self.test
        if not t.halves or mode == "exact":
            return t.src(spec)
        if mode == "snapped" and snap is not None:
            t = Test(t.how, [h.snapped(snap[0]) for h in t.halves], t.join)
        ranges = threshold_ranges(t.halves, t.join, self.X, self.want, spec)
        if ranges is None:
            if mode == "centred":
                return t.src(spec)
            ts = []
            for h in t.halves:
                lo, hi = Affine(h.w).interval(input_bounds(spec))
                ts.append(nice(h.t, snap[1] * max(hi - lo, 1e-12)))
            return t.src(spec, ts)
        if mode == "centred":
            return t.src(spec, [(lo + hi) / 2 for lo, hi in ranges])
        return t.src(spec, [nicest_in(lo, hi) for lo, hi in ranges])


def decision_list(net: SymNet, spec: Spec, mlp: MLP) -> tuple[list[tuple[str, str]], list[str]] | None:
    """([(route, program)], trace) or None when not applicable; routes: decision-list (fold constants
    as they are) and decision-list-snap (thresholds centred or snapped).

    `net` is the pruned symbolic net; `mlp` is the original, used only for an input set built like
    the checker's (uniform points, corners and points on unit and class boundaries, but another
    seed) and its decisions.
    """
    if spec.output != "argmax":
        return None
    X, _ = build_input_set(mlp, spec, seed=SAMPLE_SEED)
    y, score = net_output(mlp, spec, X)
    y = y.astype(int)
    conf = score >= spec.margin_frac * float(np.percentile(score, 95))   # the checker's unsure zone
    out = net_eval(net, X)
    classes = [int(c) for c in np.unique(y)]
    if len(classes) == 1:
        src = f"def net({', '.join(spec.arg_names)}):\n    return {classes[0]}\n"
        return [("decision-list", src)], [f"decision list: one class ({classes[0]}) on the whole box"]
    if len(classes) > MAX_CLASSES:
        return None
    names, P = _patterns(net, X)
    bool_domain = all(i.type == "bool" for i in spec.inputs)
    continuous = not any(i.type in DISCRETE_TYPES for i in spec.inputs)
    bspec = replace(spec, output="binary", n_outputs=1, threshold=0.0)
    cache: dict[tuple, Step | None] = {}

    def step(a: int, rest: frozenset, pairs: bool) -> Step | None:
        """Best test for 'class a' among the points labelled in rest."""
        key = (a, rest, pairs)
        if key in cache:
            return cache[key]
        R = np.isin(y, list(rest))
        want = y[R] == a
        others = [c for c in rest if c != a]
        sub = out[R][want][:, others]
        b = others[int(Counter(np.argmax(sub, axis=1)).most_common(1)[0][0])] if len(sub) else others[0]
        s_aff = net.outputs[a].plus(net.outputs[b].scaled(-1.0))
        s = out[R][:, a] - out[R][:, b]
        cands: list[Test] = []
        n_near = max(NEAR_MIN, int(NEAR_FRAC * R.sum()))
        near = np.argsort(np.abs(s))[:n_near]
        halves: list[Half] = []
        for pat, _ in Counter(map(tuple, P[R][near])).most_common(TOP_PATTERNS):
            h = half(fold(net, dict(zip(names, pat)), s_aff), spec)
            if h is not None and all(h.src(spec) != g.src(spec) for g in halves):
                halves.append(h)
        cands += [Test("fold", [h]) for h in halves]
        if pairs:
            for h1, h2 in itertools.combinations(halves[:PAIR_FROM], 2):
                cands += [Test("fold-or", [h1, h2], "or"), Test("fold-and", [h1, h2], "and")]
        bnet = SymNet(net.inputs, net.units, [s_aff], "binary", 0.0)
        try:
            r = boundary_expr(bnet, bspec)
        except ValueError:
            r = None
        if r is not None:
            cands.append(Test("boundary", expr=r[0], derived=r[1]))
        if continuous:
            dirs = [dict(h.w) for h in halves]
            dirs += [dict(u.pre.coef) for u in net.units if set(u.pre.coef) <= set(net.inputs) and u.pre.coef]
            dirs += [{v: 1.0} for v in net.inputs]
            r = scan_boundary_expr(lambda Z: net_eval(bnet, Z)[:, 0], bspec, dirs)
            if r is not None:
                cands.append(Test("scan", expr=r[0], derived=r[1]))
        if bool_domain:
            from .truthtable import _library
            for expr in _library(spec.arg_names):
                if np.array_equal(_holds(expr, {}, spec, X[R]), want):
                    cands.append(Test("truth-table", expr=expr))
                    break
        Xc, wc = X[R][conf[R]], want[conf[R]]
        scored = []
        for t in cands:
            ranges = threshold_ranges(t.halves, t.join, Xc, wc, spec) if t.halves else None
            ts = [(lo + hi) / 2 for lo, hi in ranges] if ranges else None
            agree = float((t.holds(Xc, spec, ts) == wc).mean()) if len(wc) else 1.0
            lines = [f"def net({', '.join(spec.arg_names)}):"]
            lines += [f"    {n} = {v}" for n, v in t.derived.items()] + [f"    return {t.src(spec)}", ""]
            size = program_size(parse_program("\n".join(lines), spec.arg_names))["nodes"]   # digits don't count
            scored.append((-agree, size, len(scored), t))
        best = Step(min(scored)[3], b, -min(scored)[0], Xc, wc) if scored else None
        cache[key] = best
        return best

    def best_list(pairs: bool) -> tuple[float, tuple, list[Step]] | None:
        found = None
        for order in itertools.permutations(classes):
            chosen = []
            for k, a in enumerate(order[:-1]):
                r = step(a, frozenset(order[k:]), pairs)
                if r is None:
                    break
                chosen.append(r)
            else:
                src = _program([(c.written(spec, "centred"), c.test.derived, a) for c, a in zip(chosen, order)],
                               order[-1], spec)
                agree = float((evaluate(parse_program(src, spec.arg_names), X[conf]) == y[conf]).mean())
                key = (-agree, len(src))
                if found is None or key < found[0]:
                    found = (key, order, chosen)
        return None if found is None else (-found[0][0], found[1], found[2])

    lists = [r for r in (best_list(False), best_list(True)) if r is not None and r[0] >= MIN_AGREE]
    if len(lists) == 2 and [c.test for c in lists[0][2]] == [c.test for c in lists[1][2]]:
        lists = lists[:1]
    if not lists:
        return None
    trace: list[str] = []
    progs: list[tuple[str, str]] = []
    for agree, order, chosen in lists:
        trace.append(f"decision list: order {list(order)}, agreement with the net on confident sample "
                     f"points {agree:.4f}")
        for c, a in zip(chosen, order):
            trace.append(f"decision list: class {a} vs class {c.other} by {c.test.how}: {c.test.src(spec)} "
                         f"(agreement {c.agree:.4f}, thresholds centred)")

        def prog(mode: str, snap=None, order=order, chosen=chosen) -> str:
            return _program([(c.written(spec, mode, snap), c.test.derived, a) for c, a in zip(chosen, order)],
                            order[-1], spec)
        variants = [("decision-list", prog("exact"))]
        variants += [("decision-list-snap", prog("snapped", tol)) for tol in SNAP_LADDER]
        variants += [("decision-list-snap", prog("centred"))]
        progs += [(r, s) for r, s in variants if s not in [p for _, p in progs]]
    return progs, trace
