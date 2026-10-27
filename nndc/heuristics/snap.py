"""P3: snap coefficients and thresholds to nice values; drop small terms only by their combined effect.

HEURISTICS: absdiff_perturbed/hand, sparse8/hand, sum8/hand, tax/seed0, xor/seed0-1, absdiff/seed0-1.
Approximate: candidates using it must pass the checker.
"""
from __future__ import annotations

from ..spec import Spec
from ..symnet import Affine, SymNet, fmt, input_bounds

NICE_DENOMS = (1, 2, 4, 5, 8, 10, 20, 50, 100)
COEF_REL = 0.05
THRESH_REL = 0.01
SMALL_REL = 0.005
SMALL_VS_MAX = 0.05


def nice(v: float, tol: float) -> float:
    if abs(v) <= tol:
        return 0.0
    for d in NICE_DENOMS:
        c = round(v * d) / d
        if abs(c - v) <= tol:
            return float(c)
    return v


def _snap_affine(a: Affine, post: dict[str, tuple[float, float]], label: str, trace: list[str]) -> Affine:
    lo, hi = a.interval(post)
    width = max(hi - lo, 1e-12)
    coef = dict(a.coef)
    # drop small terms, smallest first, while their combined effect stays negligible
    big = max((abs(c) for c in coef.values()), default=0.0)
    contrib = sorted((abs(c) * max(abs(post[v][0]), abs(post[v][1])), v) for v, c in coef.items()
                     if abs(c) < SMALL_VS_MAX * big)
    total, dropped = 0.0, []
    for eff, v in contrib:
        if total + eff > SMALL_REL * width:
            break
        total += eff
        dropped.append(v)
    for v in dropped:
        del coef[v]
    if dropped:
        trace.append(f"{label}: dropped {', '.join(dropped)} (combined effect {fmt(total)})")
    for v, c in list(coef.items()):
        s = nice(c, COEF_REL * abs(c))
        if s != c:
            trace.append(f"{label}: {v} coefficient {fmt(c)} -> {fmt(s)}")
            coef[v] = s
        if coef[v] == 0:
            del coef[v]
    const = a.const
    s1 = sum(abs(c) for c in coef.values())
    if s1 > 0:
        dlo, dhi = Affine({v: c / s1 for v, c in coef.items()}).interval(post)
        t = -const / s1
        t2 = nice(t, THRESH_REL * max(dhi - dlo, 1e-12))
        if t2 != t:
            trace.append(f"{label}: threshold {fmt(t)} -> {fmt(t2)}")
        const = -t2 * s1
    else:
        c2 = nice(const, THRESH_REL * width)
        if c2 != const:
            trace.append(f"{label}: constant {fmt(const)} -> {fmt(c2)}")
        const = c2
    return Affine(coef, const)


def snap_net(net: SymNet, spec: Spec) -> tuple[SymNet, list[str]]:
    out = net.copy()
    trace: list[str] = []
    post = input_bounds(spec)
    for u in out.units:
        u.pre = _snap_affine(u.pre, post, u.name, trace)
        lo, hi = u.pre.interval(post)
        post[u.name] = (max(lo, 0.0), max(hi, 0.0)) if u.relu else (lo, hi)
    out.outputs = [_snap_affine(o, post, f"output {k}", trace) for k, o in enumerate(out.outputs)]
    return out, trace
