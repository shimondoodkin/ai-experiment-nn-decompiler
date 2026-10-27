"""P1: dead and always-on units, from interval bounds over the input box.

HEURISTICS: deadredundant/hand, tax/seed0-1, window/seed0-1, absdiff/seed1, neareq/seed0-1, sum8/hand,
xor/seed0, ramp/hand. Exact over the declared input range.
"""
from __future__ import annotations

from ..spec import Spec
from ..symnet import Affine, SymNet, fmt, substitute, unit_bounds


def prune_dead_and_linear(net: SymNet, spec: Spec) -> tuple[SymNet, list[str]]:
    trace: list[str] = []
    changed = True
    while changed:
        changed = False
        bounds = unit_bounds(net, spec)
        for u in net.units:
            lo, hi = bounds[u.name]
            if u.relu and hi <= 0:
                trace.append(f"{u.name}: dead (pre-activation in [{fmt(lo)}, {fmt(hi)}]) -> removed")
                net = substitute(net, u.name, Affine({}, 0.0))
            elif not u.relu or lo >= 0:
                if u.relu:
                    trace.append(f"{u.name}: always on (pre-activation in [{fmt(lo)}, {fmt(hi)}]) -> linear")
                net = substitute(net, u.name, u.pre)
            else:
                continue
            changed = True
            break
    return net, trace
