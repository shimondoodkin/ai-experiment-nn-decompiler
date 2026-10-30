"""The gate view of a one-hidden-layer net: each unit as a gate line, the zones of its S-curve the box uses, a
magnitude and a sign; then the units grouped (negligible, lines, rings, same line, the rest).

Design: docs/design/2026-10-31-formula-residual-gate-view-design.md. Blind (net and blind spec only) and NumPy only.
"""
from __future__ import annotations

import numpy as np

from .model import MLP, activate, as_sigmoid
from .spec import Spec

N_SAMPLE = 100_000
N_ZGRID = 2001                       # stand-ins are compared on a grid of the unit's own z over the box
ZONES = (("off", -np.inf, -4.0), ("low knee", -4.0, -1.5), ("middle", -1.5, 1.5), ("high knee", 1.5, 4.0),
         ("on", 4.0, np.inf))
HINGE_KNOTS = 41
RING_STEP = 1.15                     # each offset within 15% of the previous one
RING_SPAN = 1.35                     # the chain's largest offset at most 1.35x its smallest
RING_MAG = 2.0                       # magnitudes within this factor of the chain's median
RING_MIN = 3
SAME_ANGLE = 10.0                    # degrees
SAME_OFFSET = 0.05                   # share of the box width


def _sample(spec: Spec) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.uniform(i.lo, i.hi, N_SAMPLE) for i in spec.inputs], axis=1)


def _one_layer(mlp: MLP) -> MLP:
    if len(mlp.weights) != 2:
        raise ValueError("the gate view needs one hidden layer")
    if mlp.n_outputs != 1:
        raise ValueError("the gate view needs one output")
    return as_sigmoid(mlp)


def tolerance(spec: Spec, mlp: MLP) -> float:
    """2% of the net's output range over the box (binary: of the raw score output - threshold, the same range)."""
    out = mlp.forward(_sample(spec).astype(np.float32))[:, 0].astype(np.float64)
    return spec.snap_tol_frac * float(out.max() - out.min())


def _hinge_err(s: np.ndarray, z: np.ndarray) -> float:
    """Max error of the best k·relu(±(z - t)) + c fit to s (least squares per knot, max error between knots)."""
    best = np.inf
    one = np.ones_like(z)
    for t in np.linspace(z.min(), z.max(), HINGE_KNOTS)[1:-1]:
        for d in (1.0, -1.0):
            A = np.stack([np.maximum(0.0, d * (z - t)), one], axis=1)
            c = np.linalg.lstsq(A, s, rcond=None)[0]
            best = min(best, float(np.abs(A @ c - s).max()))
    return best


def _stand_in(act: str, z: np.ndarray, v: float, tol: float) -> tuple[str | None, float]:
    """The simplest stand-in within a quarter of tol (errors in output units); else None and the smallest error."""
    s = activate(act, z)
    if act == "relu":
        cands = [("constant", lambda: (s.max() - s.min()) / 2),
                 ("line", lambda: np.abs(s - z).max()),
                 ("hinge", lambda: 0.0)]
    else:
        cands = [("constant", lambda: (s.max() - s.min()) / 2),
                 ("line", lambda: np.abs(s - (0.25 * z + 0.5)).max()),
                 ("step", lambda: np.abs(s - (z > 0)).max()),
                 ("clamp", lambda: np.abs(s - np.clip(0.5 + 0.25 * z, 0.0, 1.0)).max()),
                 ("rounded hinge", lambda: _hinge_err(s, z))]
    smallest = np.inf
    for name, err in cands:
        e = abs(v) * float(err())
        if e <= tol / 4:
            return name, e
        smallest = min(smallest, e)
    return None, smallest


def unit_table(spec: Spec, mlp: MLP) -> list[dict]:
    """One row per hidden unit: gate line (angle, offset, width), zone shares, magnitude, sign, stand-in."""
    net = _one_layer(mlp)
    act = net.acts[0]
    W, b = net.weights[0].astype(np.float64), net.biases[0].astype(np.float64)
    v = net.weights[1][0].astype(np.float64)
    X = _sample(spec)
    Z = X @ W.T + b
    tol = tolerance(spec, mlp)
    rows = []
    for j in range(W.shape[0]):
        w, z = W[j], Z[:, j]
        norm = float(np.linalg.norm(w))
        if len(w) == 2:
            angle = float(np.degrees(np.arctan2(w[1], w[0])))
        else:
            angle = 0.0 if w[0] >= 0 else 180.0
        if act == "relu":
            zones = {"off": float(np.mean(z <= 0)), "active": float(np.mean(z > 0))}
        else:
            zones = {name: float(np.mean((z >= lo) & (z < hi))) for name, lo, hi in ZONES}
        term = v[j] * activate(act, z)
        zg = np.linspace(float(z.min()), float(z.max()), N_ZGRID)
        stand, err = _stand_in(act, zg, float(v[j]), tol)
        rows.append({"unit": j, "angle": angle,
                     "offset": -float(b[j]) / norm if norm > 0 else float("nan"),
                     "width": (0.0 if act == "relu" else 4.0 / norm) if norm > 0 else float("inf"),
                     "zones": zones, "magnitude": float(term.max() - term.min()),
                     "sign": int(np.sign(v[j])), "stand_in": stand, "stand_in_err": err})
    return rows


def _gap_ok(angles: list[float]) -> bool:
    a = sorted(x % 360 for x in angles)
    gaps = [b - a_ for a_, b in zip(a, a[1:])] + [a[0] + 360 - a[-1]]
    return max(gaps) < 180


def _rings(units: list[dict]) -> list[list[dict]]:
    found = []
    for sign in (1, -1):
        cand = sorted((u for u in units if u["sign"] == sign and np.isfinite(u["offset"]) and u["offset"] > 0),
                      key=lambda u: (u["offset"], u["unit"]))
        i = 0
        while i < len(cand):
            chain = [cand[i]]
            k = i + 1
            while k < len(cand) and cand[k]["offset"] <= RING_STEP * chain[-1]["offset"] \
                    and cand[k]["offset"] <= RING_SPAN * chain[0]["offset"]:
                chain.append(cand[k])
                k += 1
            med = float(np.median([u["magnitude"] for u in chain]))
            ring = [u for u in chain if med / RING_MAG <= u["magnitude"] <= RING_MAG * med]
            if len(ring) >= RING_MIN and _gap_ok([u["angle"] for u in ring]):
                found.append(ring)
            i = k
    return found


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _same_line(units: list[dict], box_width: float) -> list[list[dict]]:
    n = len(units)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(n):
        for k in range(i + 1, n):
            a, b = units[i], units[k]
            d = _angle_diff(a["angle"], b["angle"])
            if d <= SAME_ANGLE:
                gap = abs(a["offset"] - b["offset"])
            elif d >= 180 - SAME_ANGLE:
                gap = abs(a["offset"] + b["offset"])       # opposite direction: the same line has offset -o
            else:
                continue
            if gap <= SAME_OFFSET * box_width:
                parent[root(k)] = root(i)
    comps: dict[int, list[dict]] = {}
    for i in range(n):
        comps.setdefault(root(i), []).append(units[i])
    return [c for c in comps.values() if len(c) >= 2]


def groups(units: list[dict], tol: float, box_width: float) -> list[dict]:
    """A partition of the units, each unit in the first group it qualifies for: negligible, lines, rings, same line,
    the rest."""
    out = []
    left = list(units)

    def take(kind: str, members: list[dict], summary: dict) -> None:
        ids = {u["unit"] for u in members}
        left[:] = [u for u in left if u["unit"] not in ids]
        out.append({"kind": kind, "units": sorted(ids), "summary": summary})

    neg = [u for u in left if u["magnitude"] < tol / 4]
    if neg:
        take("negligible", neg, {"magnitude": float(sum(u["magnitude"] for u in neg)), "count": len(neg)})
    lines = [u for u in left if u["stand_in"] in ("constant", "line")]
    if lines:
        take("lines", lines, {"count": len(lines)})
    for ring in _rings(left):
        offs = [u["offset"] for u in ring]
        take("ring", ring, {"offset_min": min(offs), "offset_max": max(offs), "count": len(ring),
                            "magnitude_median": float(np.median([u["magnitude"] for u in ring])),
                            "sign": ring[0]["sign"],
                            "stand_ins": sorted({u["stand_in"] or "none" for u in ring})})
    for comp in _same_line(left, box_width):
        take("same line", comp, {"angle": comp[0]["angle"], "offset": comp[0]["offset"], "count": len(comp)})
    if left:
        take("rest", list(left), {"count": len(left)})
    return out
