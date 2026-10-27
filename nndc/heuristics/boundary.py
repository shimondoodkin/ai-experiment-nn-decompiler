"""P6: binary outputs along one direction: find where the decision flips, snap it, emit a comparison.

HEURISTICS: neareq/seed0-1, window/seed0-1, window/hand, neareq/hand, xor/seed1. Approximate.
"""
from __future__ import annotations

import numpy as np

from ..spec import Spec
from ..symnet import Affine, SymNet, fmt, input_bounds
from .cleanup import affine_src
from .motifs import _same_dir, as_relu_sum, canonical
from .snap import nice

SNAP_REL = 0.01


def boundary_expr(net: SymNet, spec: Spec) -> tuple[str, dict[str, str]] | None:
    if spec.output != "binary":
        return None
    rs = as_relu_sum(net)
    if rs is None or not rs.terms:
        return None
    terms = [canonical(c, z) for c, z in rs.terms]
    u = Affine(dict(terms[0][1].coef), 0.0)
    m = 0.0
    if rs.linear.coef:
        k = next(iter(u.coef))
        m = rs.linear.coef.get(k, 0.0) / u.coef[k]
        if m == 0 or not _same_dir(rs.linear, u.scaled(m)):
            return None
    for _, z in terms:
        if not (_same_dir(z, u) or _same_dir(z, u, -1.0)):
            return None
    lo, hi = u.interval(input_bounds(spec))
    width = max(hi - lo, 1e-12)

    def f(y: np.ndarray) -> np.ndarray:
        out = m * y + rs.linear.const - spec.threshold
        for c, z in terms:
            s = 1.0 if _same_dir(z, u) else -1.0
            out = out + c * np.maximum(s * y + z.const, 0.0)
        return out

    knots = sorted({lo, hi} | {(-z.const if _same_dir(z, u) else z.const) for _, z in terms})
    knots = [k for k in knots if lo <= k <= hi]
    roots = []
    for a, b in zip(knots, knots[1:]):
        fa, fb = f(np.array([a, b]))
        if fa == 0 and a not in roots:
            roots.append(a)
        if fa * fb < 0:
            roots.append(a + (b - a) * fa / (fa - fb))
    roots = sorted(r for r in roots if lo < r < hi)
    if len(roots) > 2:
        return None
    edges = [lo] + roots + [hi]
    mids = np.array([(a + b) / 2 for a, b in zip(edges, edges[1:])])
    pos = list(f(mids) > 0)
    tol = SNAP_REL * width
    single = len(u.coef) == 1 and next(iter(u.coef.values())) == 1
    d = next(iter(u.coef)) if single else "d0"
    derived = {} if single else {"d0": affine_src(u)}
    if not roots:
        return ("1" if pos[0] else "0"), {}
    if len(roots) == 1:
        t = fmt(nice(roots[0], tol))
        return (f"{d} > {t}" if pos[1] else f"{d} < {t}"), derived
    a, b = roots
    if not (pos[0] == pos[2] != pos[1]):   # a band (inside positive) or its complement
        return None
    mid, r = nice((a + b) / 2, tol), nice((b - a) / 2, tol)
    centre = d if mid == 0 else f"{d} - {fmt(mid)}" if mid > 0 else f"{d} + {fmt(-mid)}"
    return (f"abs({centre}) {'<' if pos[1] else '>'} {fmt(r)}"), derived
