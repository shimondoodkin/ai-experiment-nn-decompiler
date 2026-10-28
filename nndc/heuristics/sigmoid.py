"""Sigmoid nets (project B).

Two kinds of moves:
- per-unit replacements (const, step, line, clamp), plus merging duplicate steps;
- black-box routes: linear scores and a linear cascade for argmax nets, and a knot fit for 1-input regression.

A move counts only when the program it produces passes the checker against the net.

HEURISTICS: runs_sigmoid/HEURISTICS.md (development set). The fixed move list is in
docs/plans/2026-10-28-sigmoid-nets.md ("Task 4 move list").
"""
from __future__ import annotations

import numpy as np

from ..check import build_input_set, check_program, net_output
from ..model import MLP
from ..spec import Spec
from ..symnet import Affine, SymNet, Unit, fmt, to_program, unit_value
from .snap import nice

SOFT = (0.05, 0.95)             # a sigmoid output inside this range is "soft" (the spike's saturation measure)
CONST_RANGE = 0.02              # a unit whose output moves less than this over the input set becomes a constant
SNAP_TOLS = (0.05, 0.02, 0.01, 0.005)
KNOT_GRID = 20                  # knot fit: candidate knots at lo + (hi - lo) * j / KNOT_GRID
MAX_KNOTS = 3
FIT_POINTS = 2001
NEAR_STEP = 0.05           # steps whose normalised coefficients and offset differ by at most this may merge


def _unit_outputs(net: SymNet, X: np.ndarray) -> dict[str, np.ndarray]:
    env = {n: X[:, i] for i, n in enumerate(net.inputs)}
    for u in net.units:
        env[u.name] = unit_value(u.act, u.pre.eval(env))
    return env


def soft_share(net: SymNet, X: np.ndarray) -> dict[str, float]:
    """Per sigmoid unit, the share of rows where its output is soft (not near 0 or 1)."""
    env = _unit_outputs(net, np.asarray(X, dtype=np.float64))
    return {u.name: float(((env[u.name] > SOFT[0]) & (env[u.name] < SOFT[1])).mean())
            for u in net.units if u.act == "sigmoid"}


def _replace(net: SymNet, name: str, act: str, pre: Affine) -> SymNet:
    out = net.copy()
    out.units = [Unit(u.name, u.layer, pre, act) if u.name == name else u for u in out.units]
    return out


def replace_units(net: SymNet, spec: Spec, mlp: MLP) -> tuple[SymNet, list[str]]:
    """Replace sigmoid units one at a time, most saturated first, trying const, step, line, then clamp.

    A replacement is kept only if the whole program still passes the checker against the net.
    """
    X, _ = build_input_set(mlp, spec)
    env = _unit_outputs(net, X)
    share = soft_share(net, X)
    trace: list[str] = []
    for name in sorted(share, key=lambda n: (share[n], n)):
        u = net.unit(name)
        out = env[name]
        options = []
        if float(out.max() - out.min()) < CONST_RANGE:
            options.append(("const", "linear", Affine({}, float(out.mean()))))
        options += [("step", "step", u.pre),
                    ("line", "linear", u.pre.scaled(0.25).plus(Affine({}, 0.5))),
                    ("clamp", "clamp", u.pre)]
        for label, act, pre in options:
            cand = _replace(net, name, act, pre)
            if check_program(to_program(cand), mlp, spec).snap_pass:
                net = cand
                trace.append(f"{name}: {label} (soft share {share[name]:.3f})")
                break
        else:
            trace.append(f"{name}: kept sigmoid (soft share {share[name]:.3f})")
    return net, trace


def merge_steps(net: SymNet, spec: Spec, mlp: MLP) -> tuple[SymNet, list[str]]:
    """step(k z) = step(z) for k > 0: normalise each step's largest |coef| to 1, then merge steps whose normalised
    pre-activations agree within NEAR_STEP. Each merge is kept only if the program still passes the checker."""
    out = net.copy()
    trace: list[str] = []
    for u in out.units:
        if u.act == "step" and u.pre.coef:
            u.pre = u.pre.scaled(1 / max(abs(c) for c in u.pre.coef.values()))
    merged = True
    while merged:
        merged = False
        steps = [u for u in out.units if u.act == "step"]
        for i, a in enumerate(steps):
            for b in steps[i + 1:]:
                if a.layer != b.layer or not _same(a.pre, b.pre, NEAR_STEP):
                    continue
                cand = _merge_into(out, keep=a.name, drop=b.name)
                if check_program(to_program(cand), mlp, spec).snap_pass:
                    out = cand
                    trace.append(f"{b.name}: near-duplicate step of {a.name} -> merged")
                    merged = True
                    break
            if merged:
                break
    return out, trace


def _merge_into(net: SymNet, keep: str, drop: str) -> SymNet:
    out = net.copy()
    out.units = [u for u in out.units if u.name != drop]
    for a in [u.pre for u in out.units] + out.outputs:
        c = a.coef.pop(drop, None)
        if c is not None:
            a.coef[keep] = a.coef.get(keep, 0.0) + c
    return out


def _same(a: Affine, b: Affine, tol: float = 1e-9) -> bool:
    keys = set(a.coef) | set(b.coef)
    return all(abs(a.coef.get(k, 0.0) - b.coef.get(k, 0.0)) <= tol for k in keys) and abs(a.const - b.const) <= tol


def _affine(w: np.ndarray, b: float, names: list[str]) -> Affine:
    return Affine({n: float(c) for n, c in zip(names, w) if c != 0}, float(b))


def _snapped(w: np.ndarray, b: float, tol: float) -> tuple[np.ndarray, float]:
    return np.array([nice(float(v), tol) for v in w]), nice(float(b), tol)


def _confident(mlp: MLP, spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, _ = build_input_set(mlp, spec, seed=1)
    y, score = net_output(mlp, spec, X)
    conf = score >= spec.margin_frac * float(np.percentile(score, 95))
    return X, y.astype(int), conf


def _first_passing(srcs, mlp: MLP, spec: Spec) -> str | None:
    for src in srcs:
        if check_program(src, mlp, spec).snap_pass:
            return src
    return None


def linear_scores(spec: Spec, mlp: MLP) -> str | None:
    """argmax over one linear score per class, fitted to the net's own decisions; class 0's score is 0."""
    from sklearn.linear_model import LogisticRegression

    if spec.output != "argmax":
        return None
    X, y, _ = _confident(mlp, spec)
    if len(np.unique(y)) != mlp.n_outputs:
        return None
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
    W = lr.coef_ - lr.coef_[0]
    b = lr.intercept_ - lr.intercept_[0]
    scale = float(np.abs(W).max())
    if scale == 0:
        return None
    W, b = W / scale, b / scale
    names = spec.arg_names

    def program(tol: float | None) -> str:
        lines = [f"def net({', '.join(names)}):"]
        for k in range(1, len(W)):
            w, c = (W[k], b[k]) if tol is None else _snapped(W[k], b[k], tol)
            lines.append(f"    s{k} = {_affine(w, c, names).to_src()}")
        return "\n".join(lines + [f"    return argmax(0, {', '.join(f's{k}' for k in range(1, len(W)))})"]) + "\n"

    return _first_passing([program(t) for t in SNAP_TOLS] + [program(None)], mlp, spec)


def _chain(y: np.ndarray, top2: np.ndarray, n: int) -> list[int] | None:
    """The class order a | b | c when only neighbouring classes ever are the top two."""
    pairs = {tuple(sorted(p)) for p in zip(y.tolist(), top2.tolist())}
    deg = {c: sum(c in p for p in pairs) for c in range(n)}
    ends = [c for c in range(n) if deg[c] == 1]
    if n != 3 or len(pairs) != 2 or len(ends) != 2:
        return None
    mid = next(c for c in range(n) if c not in ends)
    return [ends[0], mid, ends[1]]


def linear_cascade(spec: Spec, mlp: MLP) -> str | None:
    """For three ordered classes a | b | c: `if s_a > 0: a`, `if s_c > 0: c`, else b, fitted on confident points."""
    from sklearn.linear_model import LogisticRegression

    if spec.output != "argmax" or mlp.n_outputs != 3:
        return None
    X, y, conf = _confident(mlp, spec)
    out = mlp.forward(X.astype(np.float32)).astype(np.float64)
    top2 = np.argsort(out, axis=1)[:, -2]
    order = _chain(y[conf], top2[conf], 3)
    if order is None:
        return None
    a, mid, c = order
    cuts = []
    for cls in (a, c):
        lr = LogisticRegression(C=1e4, max_iter=5000).fit(X[conf], (y[conf] == cls).astype(int))
        w, b = lr.coef_[0], float(lr.intercept_[0])
        s = float(np.abs(w).max())
        if s == 0:
            return None
        cuts.append((w / s, b / s))
    names = spec.arg_names

    def program(tol: float | None) -> str:
        lines = [f"def net({', '.join(names)}):"]
        for cls, (w, b) in zip((a, c), cuts):
            w2, b2 = (w, b) if tol is None else _snapped(w, b, tol)
            lines.append(f"    s{cls} = {_affine(w2, b2, names).to_src()}")
            lines.append(f"    if s{cls} > 0:\n        return {cls}")
        return "\n".join(lines + [f"    return {mid}"]) + "\n"

    return _first_passing([program(t) for t in SNAP_TOLS] + [program(None)], mlp, spec)


def _minimax(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """min_c max |A c - y| by a linear program; returns (c, max error)."""
    from scipy.optimize import linprog

    n, k = A.shape
    cost = np.zeros(k + 1)
    cost[-1] = 1.0
    ones = np.ones((n, 1))
    A_ub = np.vstack([np.hstack([A, -ones]), np.hstack([-A, -ones])])
    b_ub = np.concatenate([y, -y])
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * k + [(0, None)], method="highs")
    if r.status != 0:
        return np.zeros(k), float("inf")
    return r.x[:k], float(r.x[-1])


def knot_fit(spec: Spec, mlp: MLP) -> str | None:
    """1-input regression: c0 + s0 x + sum_k (s_k - s_{k-1}) relu(x - knot_k), knots on a grid, minimax fit."""
    if spec.output != "regression" or mlp.n_inputs != 1:
        return None
    lo, hi = float(spec.inputs[0].lo), float(spec.inputs[0].hi)
    x = np.linspace(lo, hi, FIT_POINTS)
    y = mlp.forward(x[:, None].astype(np.float32))[:, 0].astype(np.float64)
    grid = [lo + (hi - lo) * j / KNOT_GRID for j in range(1, KNOT_GRID)]
    name = spec.arg_names[0]

    def design(knots: list[float]) -> np.ndarray:
        return np.column_stack([np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in knots])

    def program(knots: list[float], slopes: list[float], c0: float) -> str:
        expr = Affine({name: slopes[0]}, c0).to_src()
        for k, (s_prev, s) in zip(knots, zip(slopes, slopes[1:])):
            d = s - s_prev
            if d != 0:
                expr += f" {'-' if d < 0 else '+'} {fmt(abs(d))} * relu({Affine({name: 1.0}, -k).to_src()})"
        return f"def net({name}):\n    return {expr}\n"

    knots: list[float] = []
    for _ in range(MAX_KNOTS):
        best = min((g for g in grid if g not in knots), key=lambda g: _minimax(design(sorted(knots + [g])), y)[1])
        knots = sorted(knots + [best])
        c, _ = _minimax(design(knots), y)
        slopes = list(np.cumsum(c[1:]))
        tol = 0.05 * max(abs(s) for s in slopes) if any(slopes) else 0.0
        snapped = [nice(float(s), tol) for s in slopes]
        hinge = np.column_stack([x] + [np.maximum(x - k, 0.0) for k in knots])
        rest = y - hinge @ np.array([snapped[0]] + [b - a for a, b in zip(snapped, snapped[1:])])
        c0 = float((rest.max() + rest.min()) / 2)
        src = _first_passing([program(knots, snapped, c0), program(knots, [float(s) for s in slopes], float(c[0]))],
                             mlp, spec)
        if src is not None:
            return src
    return None
