"""P5: read a one-direction net as a 1-D spline (kinks and slopes); convex -> max(lines), concave -> min(lines).

HEURISTICS: tax/seed0-1, triangle/hand, ramp/hand, window/seed0, absdiff/seed0. Approximate.
"""
from __future__ import annotations

from ..spec import Spec
from ..symnet import Affine, input_bounds
from .cleanup import affine_src
from .motifs import ReluSum, _close, _same_dir, canonical
from .snap import nice

MERGE_REL = 0.01
KINK_REL = 0.01
SLOPE_REL = 0.05


def spline_expr(rs: ReluSum, spec: Spec) -> str | None:
    if rs is None or not rs.terms:
        return None
    terms = [canonical(c, z) for c, z in rs.terms]
    u = Affine(dict(terms[0][1].coef), 0.0)
    # linear part must lie along u
    m = 0.0
    if rs.linear.coef:
        k = next(iter(u.coef))
        m = rs.linear.coef.get(k, 0.0) / u.coef[k]
        if not _same_dir(rs.linear.scaled(1.0), u.scaled(m)) or m == 0:
            return None
    b = rs.linear.const
    lo, hi = u.interval(input_bounds(spec))
    width = max(hi - lo, 1e-12)
    hinges: list[tuple[float, float]] = []          # (kink y0, slope change to the right)
    for c, z in terms:
        if _same_dir(z, u):
            y0 = -z.const
            hinges.append((y0, c))
        elif _same_dir(z, u, -1.0):                  # c relu(-(y - y0)) = c relu(y - y0) - c (y - y0)
            y0 = z.const
            hinges.append((y0, c))
            m -= c
            b += c * y0
        else:
            return None
    # fold hinges outside the range
    kept = []
    for y0, d in hinges:
        if y0 <= lo:
            m += d
            b -= d * y0
        elif y0 < hi:
            kept.append((y0, d))
    kept.sort()
    merged: list[list[float]] = []
    for y0, d in kept:
        if merged and y0 - merged[-1][0] <= MERGE_REL * width:
            py, pd = merged[-1]
            w = abs(pd) + abs(d)
            merged[-1] = [(py * abs(pd) + y0 * abs(d)) / w if w else py, pd + d]
        else:
            merged.append([y0, d])
    merged = [h for h in merged if not _close(h[1], 0.0)]
    if not merged:
        return None
    f_lo = m * lo + b
    slopes = [m]
    for _, d in merged:
        slopes.append(slopes[-1] + d)
    slopes = [nice(s, SLOPE_REL * abs(s)) for s in slopes]
    kinks = [nice(y0, KINK_REL * width) for y0, _ in merged]
    if all(s1 <= s2 for s1, s2 in zip(slopes, slopes[1:])):
        fn = "max"
    elif all(s1 >= s2 for s1, s2 in zip(slopes, slopes[1:])):
        fn = "min"
    else:
        return None
    out_range = max(abs(s) for s in slopes) * width or 1.0
    b0 = nice(f_lo - slopes[0] * lo, KINK_REL * out_range)
    lines = []
    bi = b0
    for i, s in enumerate(slopes):
        if i > 0:
            bi = bi + (slopes[i - 1] - s) * kinks[i - 1]
        lines.append(affine_src(Affine({v: s * c for v, c in u.coef.items() if s * c != 0}, round(bi, 10))))
    return f"{fn}({', '.join(lines)})"
