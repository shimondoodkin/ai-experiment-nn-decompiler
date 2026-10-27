"""B4: exact activation-region decomposition, emitted as a nested if/else with an affine leaf per region.

Depth-first over hidden units in order; inside a region each unit's pre-activation is affine in the
inputs, and an LP (max slack, HiGHS) keeps only the sides with a non-empty interior. Exact by
construction; the number of regions is what it costs.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..model import MLP
from ..run import load_run, record_git
from ..spec import Spec
from .common import write_info

MAX_REGIONS = 4096
MAX_SECONDS = 60.0
MIN_SLACK = 1e-9


class _Budget(Exception):
    pass


def _affine_src(a: np.ndarray, c: float, names: list[str]) -> str:
    parts = []
    for coef, name in zip(a, names):
        if coef == 0:
            continue
        term = name if abs(coef) == 1 else f"{abs(float(coef))!r} * {name}"
        parts.append(("- " if coef < 0 else "+ ") + term)
    if c != 0 or not parts:
        parts.append(("- " if c < 0 else "+ ") + repr(abs(float(c))))
    s = " ".join(parts)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


def regions_program(mlp: MLP, spec: Spec, max_regions: int = MAX_REGIONS,
                    max_seconds: float = MAX_SECONDS) -> tuple[str | None, dict]:
    from scipy.optimize import linprog

    t0 = time.perf_counter()
    lo = np.array([i.lo for i in spec.inputs], dtype=np.float64)
    hi = np.array([i.hi for i in spec.inputs], dtype=np.float64)
    d = lo.size
    Ws = [W.astype(np.float64) for W in mlp.weights]
    bs = [b.astype(np.float64) for b in mlp.biases]
    stats = {"regions": 0, "lp_calls": 0}
    bounds = [(float(a), float(b)) for a, b in zip(lo, hi)] + [(None, 1.0)]
    obj = np.zeros(d + 1)
    obj[-1] = -1.0

    def interior(cons: list[tuple[np.ndarray, float]]) -> bool:
        """Is {x in box : g.x + h > 0 for all (g, h)} non-empty with an interior?"""
        rows, rhs = [], []
        for g, h in cons:
            n = np.linalg.norm(g)
            if n == 0:
                if h <= 0:
                    return False
                continue
            rows.append(np.append(-g / n, 1.0))   # -g.x/n + s <= h/n
            rhs.append(h / n)
        if not rows:
            return True
        stats["lp_calls"] += 1
        r = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")
        return r.status == 0 and -r.fun > MIN_SLACK

    def rec(layer: int, j: int, PA: np.ndarray, pc: np.ndarray, HA: np.ndarray, hc: np.ndarray,
            cons: list) -> tuple:
        if time.perf_counter() - t0 > max_seconds:
            raise _Budget
        if layer == len(Ws) - 1:   # output layer: one leaf per region
            stats["regions"] += 1
            if stats["regions"] > max_regions:
                raise _Budget
            return ("leaf", Ws[layer] @ PA, Ws[layer] @ pc + bs[layer])
        ZA, zc = Ws[layer] @ PA, Ws[layer] @ pc + bs[layer]
        if j == ZA.shape[0]:
            return rec(layer + 1, 0, HA, hc, np.zeros((0, d)), np.zeros(0), cons)
        a, c = ZA[j], float(zc[j])
        pos = interior(cons + [(a, c)])
        neg = interior(cons + [(-a, -c)])
        on = lambda cs: rec(layer, j + 1, PA, pc, np.vstack([HA, a]), np.append(hc, c), cs)  # noqa: E731
        off = lambda cs: rec(layer, j + 1, PA, pc, np.vstack([HA, np.zeros(d)]), np.append(hc, 0.0), cs)  # noqa: E731
        if pos and neg:
            return ("split", a, c, on(cons + [(a, c)]), off(cons + [(-a, -c)]))
        return on(cons) if pos else off(cons)

    try:
        tree = rec(0, 0, np.eye(d), np.zeros(d), np.zeros((0, d)), np.zeros(0), [])
    except _Budget:
        return None, {**stats, "budget_exceeded": True, "seconds": round(time.perf_counter() - t0, 2)}
    names = spec.arg_names

    def emit(node: tuple, depth: int) -> list[str]:
        pad = "    " * depth
        if node[0] == "split":
            return ([f"{pad}if {_affine_src(node[1], node[2], names)} > 0:"] + emit(node[3], depth + 1)
                    + [f"{pad}else:"] + emit(node[4], depth + 1))
        outs = [_affine_src(A, c, names) for A, c in zip(node[1], node[2])]
        body = f"argmax({', '.join(outs)})" if spec.output == "argmax" else outs[0]
        return [f"{pad}return {body}"]

    src = f"def net({', '.join(names)}):\n" + "\n".join(emit(tree, 1)) + "\n"
    return src, {**stats, "budget_exceeded": False, "seconds": round(time.perf_counter() - t0, 2)}


def run_regions(rd: Path) -> dict:
    spec_full, mlp = load_run(rd)
    spec = Spec.from_json(spec_full.blind())
    src, info = regions_program(mlp, spec)
    if spec.is_discrete() or any(i.type != "float" for i in spec.inputs):
        info["note"] = "integer inputs: LP on the real relaxation; some regions may hold no integer point"
    path = Path(rd) / "regions.py"
    if src is not None:
        path.write_text(src, encoding="utf-8")
    elif path.exists():
        path.unlink()
    write_info(rd, "regions", info)
    record_git(rd, "regions")
    return info
