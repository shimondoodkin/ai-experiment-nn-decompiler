"""P6b: measure a binary net's decision boundary directly along a candidate direction.

Scan lines across the input box, find where the decision flips, and check that the flips sit at the
same d = u.x on every line ("if the boundary barely moves, it is a constant threshold").
HEURISTICS: neareq/seed0-1, window/seed0-1. Approximate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..spec import DISCRETE_TYPES, Spec
from ..symnet import Affine, input_bounds
from .boundary import emit_rule, positive_leading
from .snap import nice

N_LINES = 30
N_STEPS = 400
BISECT = 30
CLUSTER_GAP = 0.02
CLUSTER_SPREAD = 0.04
SNAP_REL = 0.0025          # snap tolerance floor; otherwise the measured spread of the crossings
MIN_AGREE = 0.97


def _snap_direction(coef: dict[str, float], spec: Spec) -> Affine | None:
    m = max((abs(c) for c in coef.values()), default=0.0)
    if m == 0:
        return None
    snapped = {v: nice(c / m, 0.05 * abs(c / m)) for v, c in coef.items()}
    snapped = {v: c for v, c in snapped.items() if c != 0}
    return positive_leading(Affine(snapped, 0.0), spec) if snapped else None


def _crossings(score: Callable, base: np.ndarray, v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[float]:
    vv = float(v @ v)
    step = v / vv                                  # moving t along the line changes d = u.x by t
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(step != 0, (lo - base) / step, -np.inf)
        t2 = np.where(step != 0, (hi - base) / step, np.inf)
    tmin, tmax = np.max(np.minimum(t1, t2)), np.min(np.maximum(t1, t2))
    if not tmin < tmax:
        return []
    t = np.linspace(tmin, tmax, N_STEPS)
    s = score(base + t[:, None] * step) > 0
    out = []
    for k in np.nonzero(s[:-1] != s[1:])[0]:
        a, b = t[k], t[k + 1]
        sa = s[k]
        for _ in range(BISECT):
            mid = (a + b) / 2
            if (score((base + mid * step)[None, :])[0] > 0) == sa:
                a = mid
            else:
                b = mid
        out.append(float((a + b) / 2))
    return out


def scan_boundary_expr(score: Callable[[np.ndarray], np.ndarray], spec: Spec,
                       directions: list[dict[str, float]]) -> tuple[str, dict[str, str]] | None:
    if spec.output != "binary" or any(i.type in DISCRETE_TYPES for i in spec.inputs):
        return None
    names = spec.arg_names
    b = input_bounds(spec)
    lo = np.array([b[n][0] for n in names])
    hi = np.array([b[n][1] for n in names])
    rng = np.random.default_rng(0)
    seen = set()
    for coef in directions:
        u = _snap_direction(coef, spec)
        if u is None:
            continue
        key = tuple(sorted(u.coef.items()))
        if key in seen:
            continue
        seen.add(key)
        v = np.array([u.coef.get(n, 0.0) for n in names])
        dlo, dhi = u.interval(b)
        width = max(dhi - dlo, 1e-12)
        ds = []
        for base in rng.uniform(lo, hi, (N_LINES, len(names))):
            d0 = float(v @ base)
            ds += [d0 + t for t in _crossings(score, base, v, lo, hi)]
        if not ds:
            continue
        ds.sort()
        clusters = [[ds[0]]]
        for d in ds[1:]:
            if d - clusters[-1][-1] > CLUSTER_GAP * width:
                clusters.append([d])
            else:
                clusters[-1].append(d)
        if len(clusters) > 2 or any(c[-1] - c[0] > CLUSTER_SPREAD * width for c in clusters):
            continue
        roots = [float(np.median(c)) for c in clusters]
        margins = [max(c[-1] - c[0], 0.01 * width) for c in clusters]
        X = rng.uniform(lo, hi, (5000, len(names)))
        d = X @ v
        dec = score(X) > 0
        near = np.zeros(len(d), bool)              # points where the boundary itself wobbles don't vote
        for r, mg in zip(roots, margins):
            near |= np.abs(d - r) < mg
        edges = [-np.inf] + roots + [np.inf]
        pos, ok = [], True
        for a, c in zip(edges, edges[1:]):
            sel = (d > a) & (d < c) & ~near
            if sel.sum() < 50:
                ok = False
                break
            frac = dec[sel].mean()
            pos.append(bool(frac > 0.5))
            ok &= max(frac, 1 - frac) >= MIN_AGREE
        if not ok:
            continue
        tol = max(max(c[-1] - c[0] for c in clusters), SNAP_REL * width)   # measurement precision
        rule = emit_rule(u, roots, pos, tol)
        if rule is not None:
            return rule
    return None
