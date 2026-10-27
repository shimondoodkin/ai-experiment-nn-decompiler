"""P2: normalize units to unit L1 direction and merge duplicates (exact) or near-duplicates (approximate).

HEURISTICS: absdiff/seed0-1, neareq/seed0-1, deadredundant/hand, median3/seed1, xor/seed1, window/seed0.
"""
from __future__ import annotations

import math

from ..spec import Spec
from ..symnet import Affine, SymNet, fmt, input_bounds, unit_bounds

NEAR_COS = 0.98
NEAR_DT = 0.02


def _rescale_consumers(net: SymNet, name: str, k: float) -> None:
    for u in net.units:
        if name in u.pre.coef:
            u.pre.coef[name] *= k
    for o in net.outputs:
        if name in o.coef:
            o.coef[name] *= k


def normalize(net: SymNet) -> SymNet:
    """relu(a.x + b) = s * relu(a/s . x + b/s) with s = |a|_1; consumers get the factor s (exact)."""
    out = net.copy()
    for u in out.units:
        s = sum(abs(c) for c in u.pre.coef.values())
        if u.relu and s > 0 and s != 1:
            u.pre = u.pre.scaled(1 / s)
            _rescale_consumers(out, u.name, s)
    return out


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(c * b.get(v, 0.0) for v, c in a.items())
    na = math.sqrt(sum(c * c for c in a.values()))
    nb = math.sqrt(sum(c * c for c in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _same(a: Affine, b: Affine) -> bool:
    keys = set(a.coef) | set(b.coef)
    return all(abs(a.coef.get(k, 0.0) - b.coef.get(k, 0.0)) <= 1e-12 for k in keys) and abs(a.const - b.const) <= 1e-12


def _out_weight(net: SymNet, name: str) -> float:
    return sum(abs(u.pre.coef.get(name, 0.0)) for u in net.units) + sum(abs(o.coef.get(name, 0.0)) for o in net.outputs)


def merge_units(net: SymNet, spec: Spec, near: bool) -> tuple[SymNet, list[str]]:
    out = net.copy()
    trace: list[str] = []
    post = input_bounds(spec)
    pre_b = unit_bounds(out, spec)
    for u in out.units:
        lo, hi = pre_b[u.name]
        post[u.name] = (max(lo, 0.0), max(hi, 0.0)) if u.relu else (lo, hi)
    merged = True
    while merged:
        merged = False
        units = [u for u in out.units if u.relu and u.pre.coef]
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                if a.layer != b.layer:
                    continue
                if _same(a.pre, b.pre):
                    how = "duplicate"
                elif near and _cos(a.pre.coef, b.pre.coef) >= NEAR_COS:
                    lo, hi = Affine(a.pre.coef).interval(post)
                    if abs(a.pre.const - b.pre.const) > NEAR_DT * max(hi - lo, 1e-12):
                        continue
                    how = "near-duplicate"
                else:
                    continue
                wa, wb = _out_weight(out, a.name), _out_weight(out, b.name)
                if how == "near-duplicate" and wa + wb > 0:
                    a.pre = a.pre.scaled(wa / (wa + wb)).plus(b.pre.scaled(wb / (wa + wb)))
                for cons in [x.pre for x in out.units] + out.outputs:
                    if b.name in cons.coef:
                        cons.coef[a.name] = cons.coef.get(a.name, 0.0) + cons.coef.pop(b.name)
                        if cons.coef[a.name] == 0:
                            del cons.coef[a.name]
                out.units = [x for x in out.units if x.name != b.name]
                trace.append(f"{b.name} merged into {a.name} ({how}, threshold {fmt(-a.pre.const)})")
                merged = True
                break
            if merged:
                break
    return out, trace
