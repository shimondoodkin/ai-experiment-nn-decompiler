"""P4: exact ReLU motifs (abs, passthrough, hat, clamp, max/min) on single-hidden-layer nets.

HEURISTICS: max/hand, deadredundant/hand, absdiff/hand, ramp/hand, triangle/hand, window/hand, neareq/hand,
sum8/hand, xor/hand.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..symnet import Affine, SymNet, fmt
from .cleanup import affine_src, join_terms, var_items

EPS = 1e-9


@dataclass
class ReluSum:
    terms: list[tuple[float, Affine]]   # c * relu(z)
    linear: Affine


def as_relu_sum(net: SymNet, out: int = 0) -> ReluSum | None:
    if any(u.act in ("sigmoid", "tanh") for u in net.units):
        return None
    inputs = set(net.inputs)
    if any(not set(u.pre.coef) <= inputs for u in net.units):
        return None
    o = net.outputs[out]
    lin = Affine({v: c for v, c in o.coef.items() if v in inputs}, o.const)
    terms = []
    for v, c in o.coef.items():
        if v in inputs:
            continue
        u = net.unit(v)
        if u.act == "linear":
            lin = lin.plus(u.pre.scaled(c))
        elif not u.pre.coef:
            lin = lin.plus(Affine({}, c * max(u.pre.const, 0.0)))
        else:
            terms.append((c, u.pre))
    return ReluSum(terms, lin)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= EPS * max(1.0, abs(a), abs(b))


def _same_dir(a: Affine, b: Affine, sign: float = 1.0) -> bool:
    keys = set(a.coef) | set(b.coef)
    return all(_close(a.coef.get(k, 0.0), sign * b.coef.get(k, 0.0)) for k in keys)


def canonical(c: float, z: Affine) -> tuple[float, Affine]:
    """c * relu(z) rescaled so that the largest |coefficient| of z is 1 (relu is positively homogeneous)."""
    m = max((abs(k) for k in z.coef.values()), default=1.0)
    return c * m, z.scaled(1 / m)


def _thr(z: Affine) -> float:
    return -z.const


def motif_expr(rs: ReluSum) -> str:
    terms = [list(canonical(c, z)) for c, z in rs.terms]
    lin = rs.linear
    pieces: list[tuple[float, str]] = []

    # opposite pairs: c relu(z) - c relu(-z) = c z ; c relu(z) + c relu(-z) = c abs(z)
    found = True
    while found:
        found = False
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                (ci, zi), (cj, zj) = terms[i], terms[j]
                if _same_dir(zi, zj, -1.0) and _close(zi.const, -zj.const):
                    if _close(ci, -cj):
                        lin = lin.plus(zi.scaled(ci))
                    elif _close(ci, cj):
                        pieces.append((ci, f"abs({affine_src(zi)})"))
                    else:
                        continue
                    terms = [t for k, t in enumerate(terms) if k not in (i, j)]
                    found = True
                    break
            if found:
                break

    # hat: a, -2a, a on equally spaced thresholds t-d, t, t+d along one direction
    found = True
    while found:
        found = False
        for i in range(len(terms)):
            group = [k for k in range(len(terms)) if _same_dir(terms[k][1], terms[i][1])]
            if len(group) < 3:
                continue
            g = sorted(group, key=lambda k: _thr(terms[k][1]))
            for a, b, c in zip(g, g[1:], g[2:]):
                ta, tb, tc = (_thr(terms[k][1]) for k in (a, b, c))
                ca, cb, cc = (terms[k][0] for k in (a, b, c))
                if _close(ca, cc) and _close(cb, -2 * ca) and _close(tb - ta, tc - tb):
                    d = tb - ta
                    y = Affine(dict(terms[b][1].coef), 0.0)
                    inner = f"abs({affine_src(y.plus(Affine({}, -tb)))})"
                    inner = inner if _close(d, 1.0) else f"{inner} / {fmt(d)}"
                    pieces.append((ca * d, f"max(0, 1 - {inner})"))
                    terms = [t for k, t in enumerate(terms) if k not in (a, b, c)]
                    found = True
                    break
            if found:
                break

    # clamp: c relu(y - t1) - c relu(y - t2), t1 < t2  ->  c * min(max(y - t1, 0), t2 - t1)
    found = True
    while found:
        found = False
        for i in range(len(terms)):
            for j in range(len(terms)):
                if i == j:
                    continue
                (ci, zi), (cj, zj) = terms[i], terms[j]
                if _same_dir(zi, zj) and _close(ci, -cj) and _thr(zi) < _thr(zj) - EPS:
                    y = Affine(dict(zi.coef), 0.0)
                    lo = affine_src(y.plus(Affine({}, -_thr(zi))))
                    pieces.append((ci, f"min(max({lo}, 0), {fmt(_thr(zj) - _thr(zi))})"))
                    terms = [t for k, t in enumerate(terms) if k not in (i, j)]
                    found = True
                    break
            if found:
                break

    # one relu left and nothing else: L + c relu(z) = max(L, L + c z) (c > 0) or min(...) (c < 0)
    if len(terms) == 1 and not pieces:
        c, z = terms[0]
        a, b = affine_src(lin), affine_src(lin.plus(z.scaled(c)))
        return f"{'max' if c > 0 else 'min'}({', '.join(sorted([a, b]))})"

    for c, z in terms:
        pieces.append((c, f"relu({affine_src(z)})"))
    return join_terms(var_items(lin) + [(lin.const, None)] + pieces)
