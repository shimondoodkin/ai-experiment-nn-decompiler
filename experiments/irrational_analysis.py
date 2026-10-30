"""Irrational runs: how the net models the function, and whether ast5 found the formula (report only).

Task-aware on purpose: it reads the true target and the task's family, which the tool never sees. One run's analysis
is written to irr_analysis.json; nndc/irrational_report.py only reads those files.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

from nndc.check import build_input_set, check_program, net_output
from nndc.heuristics.cleanup import join_terms, program_from_expr
from nndc.lang import parse_program, program_size
from nndc.model import MLP
from nndc.spec import Spec

from .irrational import FAMILY
from .tasks import get_task

ANALYSIS = "irr_analysis.json"
MAX_TERMS = 4
MERGE_FRAC = 0.01        # kinks closer than this share of the box are one kink
EDGE_FRAC = 0.05
MAX_HINGES = 16
KNOTS = 97
N_GRID = 2001
N_RAYS = 720
CENTRE = {"circle": (0.0, 0.0), "ellipse": (0.5, 0.0)}   # 2-D regression tasks are all centred at the origin
FAMILY_TOKENS = {"exp": r"\bexp\(", "log": r"\blog\(", "sqrt": r"\bsqrt\(", "reciprocal": r"/ *\(?[^0-9 ]",
                 "square": r"x0 *\* *x0", "quadratic": r"x\d *\* *x\d", "conic": r"x\d *\* *x\d"}


def formula_found(rows: list[dict], family: str) -> dict:
    """From ast5's family rows: was the task's family found (passing, at most 4 correction terms)?"""
    passing = [r for r in rows if r.get("passed")]
    mine = [r for r in passing if r["family"] == family and r.get("terms", 0) <= MAX_TERMS]
    pick = min(passing, key=lambda r: (r.get("nodes") or 10**9, r.get("terms", 0)), default=None)
    return {"formula_found": bool(mine),
            "found_terms": min(r.get("terms", 0) for r in mine) if mine else None,
            "formula_nodes": min(r["nodes"] for r in mine) if mine else None,
            "other_families": sorted({r["family"] for r in passing if r["family"] not in (family, "identity")}),
            "route_pick": pick["family"] if pick else None}


def near_fields(near: dict | None, family: str) -> dict:
    """ast5's near formula (recorded when no formula passes): its family, error, terms, and whether it is the task's
    family."""
    if not near:
        return {"near_family": None, "near_err": None, "near_terms": None, "near_family_is_task_family": None}
    return {"near_family": near["family"], "near_err": near["err"], "near_terms": near["terms"],
            "near_family_is_task_family": near["family"] == family}


def merge_kinks(xs: list[float], width: float) -> list[float]:
    """Sorted kinks; neighbours closer than MERGE_FRAC of the width become one (their mean, 3 digits)."""
    out: list[list[float]] = []
    for x in sorted(xs):
        if out and x - out[-1][-1] <= MERGE_FRAC * width:
            out[-1].append(x)
        else:
            out.append([x])
    return [round(float(np.mean(g)), 3) for g in out]


def kinks_1d(mlp: MLP, lo: float, hi: float) -> dict:
    """Kinks (ReLU) or centres (sigmoid, tanh) of a one-layer, one-input net inside the box; dead ReLU units are
    never active in the box, or have zero output weight."""
    W1, b1, W2 = mlp.weights[0][:, 0].astype(float), mlp.biases[0].astype(float), mlp.weights[1][0].astype(float)
    pts, dead = [], 0
    for w, b, o in zip(W1, b1, W2):
        on = [w * x + b > 0 for x in (lo, hi)]
        if mlp.acts[0] == "relu" and (o == 0 or not any(on)):
            dead += 1
            continue
        if w != 0 and lo < -b / w < hi:
            pts.append(float(-b / w))
    key = "kinks" if mlp.acts[0] == "relu" else "centres"
    return {key: merge_kinks(pts, hi - lo), "dead_units": dead if mlp.acts[0] == "relu" else None}


def optimal_knots(f: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, n: int) -> list[float]:
    """n knots at the quantiles (i + 1/2)/n of the density sqrt|f''| (numeric second differences)."""
    if n <= 0:
        return []
    x = np.linspace(lo, hi, 4001)
    with np.errstate(all="ignore"):
        y = f(x)
    d2 = np.abs(np.gradient(np.gradient(y, x), x))
    dens = np.sqrt(np.nan_to_num(d2, nan=0.0, posinf=0.0))
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(x))])
    if cdf[-1] <= 0:
        return [round(float(v), 3) for v in np.linspace(lo, hi, n + 2)[1:-1]]
    return [round(float(np.interp((i + 0.5) / n * cdf[-1], cdf, x)), 3) for i in range(n)]


def knot_gap(kinks: list[float], optimal: list[float], width: float) -> float | None:
    if not kinks or len(kinks) != len(optimal):
        return None
    return round(float(np.mean(np.abs(np.sort(kinks) - np.sort(optimal)))) / width, 4)


def won_on_size(formula_nodes: int | None, hinge_nodes: int | None, ast4_nodes: int | None) -> bool:
    if formula_nodes is None:
        return False
    return all(formula_nodes < n for n in (hinge_nodes, ast4_nodes) if n is not None)


def pysr_found(src: str | None, family: str) -> bool:
    if not src:
        return False
    body = src.split("\n", 1)[1] if "\n" in src else src
    return re.search(FAMILY_TOKENS[family], body) is not None


def hinge_fit(spec: Spec, mlp: MLP) -> dict:
    """The from-scratch hinge program (spike): c0 + c1·x0 + Σ k·relu(x0 − t), greedy knots, each step checked."""
    i = spec.inputs[0]
    x = np.linspace(i.lo, i.hi, N_GRID)
    y = net_output(mlp, spec, x[:, None])[0].astype(float)
    knots = np.linspace(i.lo, i.hi, KNOTS)[1:-1]
    cols, chosen = [np.ones_like(x), x], []
    for k in range(MAX_HINGES + 1):
        c = np.linalg.lstsq(np.stack(cols, 1), y, rcond=None)[0]
        items = [(float(f"{c[1]:.4g}"), "x0")]
        items += [(float(f"{ck:.4g}"), f"relu({join_terms([(1.0, 'x0'), (-t, None)])})")
                  for ck, t in zip(c[2:], chosen)]
        src = program_from_expr(join_terms(items + [(float(f"{c[0]:.4g}"), None)]), spec)
        if check_program(src, mlp, spec).snap_pass:
            return {"hinges": k, "nodes": program_size(parse_program(src, spec.arg_names))["nodes"], "passed": True}
        if k == MAX_HINGES:
            break
        best = None
        for t in knots:
            col = np.maximum(0.0, x - t)
            A = np.stack(cols + [col], 1)
            e = float(np.abs(A @ np.linalg.lstsq(A, y, rcond=None)[0] - y).max())
            if best is None or e < best[0]:
                best = (e, float(f"{t:.4g}"), col)
        chosen.append(best[1])
        cols.append(best[2])
    return {"hinges": None, "nodes": None, "passed": False}


def _boundary_rays(spec: Spec, mlp: MLP, centre: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Per ray from the centre, the first point where the net's decision flips inside the box (NaN if none)."""
    cx, cy = centre
    lo = np.array([i.lo for i in spec.inputs])
    hi = np.array([i.hi for i in spec.inputs])
    rmax = float(np.linalg.norm(hi - lo))
    pts = np.full((N_RAYS, 2), np.nan)
    for k, th in enumerate(np.linspace(0, 2 * math.pi, N_RAYS, endpoint=False)):
        d = np.array([math.cos(th), math.sin(th)])
        r = np.linspace(0, rmax, 400)
        P = np.array([cx, cy]) + r[:, None] * d
        inbox = ((P >= lo) & (P <= hi)).all(1)
        dec = net_output(mlp, spec, P)[0]
        flip = np.nonzero((dec[:-1] != dec[1:]) & inbox[1:])[0]
        if not len(flip):
            continue
        a, b = r[flip[0]], r[flip[0] + 1]
        da = dec[flip[0]]
        for _ in range(40):
            m = (a + b) / 2
            if net_output(mlp, spec, (np.array([cx, cy]) + m * d)[None])[0][0] == da:
                a = m
            else:
                b = m
        pts[k] = np.array([cx, cy]) + (a + b) / 2 * d
    ok = ~np.isnan(pts[:, 0])
    return pts[ok], np.linalg.norm(pts[ok] - np.array([cx, cy]), axis=1)


def _model_2d_binary(task: str, spec: Spec, mlp: MLP, rows: list[dict]) -> dict:
    pts, radius = _boundary_rays(spec, mlp, CENTRE[task])
    pieces = None
    if len(pts) and mlp.acts[0] == "relu":
        pre = pts.astype(np.float32) @ mlp.weights[0].T + mlp.biases[0]
        pieces = len({tuple(p) for p in (pre > 0)})
    conic = next((r.get("snapped") for r in rows if r["family"] == "conic"), None)
    return {"pieces": pieces, "boundary_points": int(len(pts)), "conic": conic,
            "radius_min_max": [round(float(radius.min()), 3), round(float(radius.max()), 3)] if len(pts) else None}


def _model_2d_regression(spec: Spec, mlp: MLP, target) -> dict:
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(i.lo, i.hi, 20_000) for i in spec.inputs], axis=1)
    err = np.abs(net_output(mlp, spec, X)[0] - target(X))
    r = np.linalg.norm(X, axis=1)
    edges = np.linspace(0, r.max(), 11)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r <= b)
        out.append(round(float(err[m].max()), 4) if m.any() else None)
    return {"err_by_radius": out, "radius_max": round(float(r.max()), 3)}


def analyse_run(rd: Path) -> dict:
    from nndc.run import load_run, record_git

    rd = Path(rd)
    spec, mlp = load_run(rd)
    task = get_task(spec.task)
    family = FAMILY[spec.task]
    res = json.loads((rd / "results.json").read_text(encoding="utf-8"))["versions"]
    log = json.loads((rd / "train_log.json").read_text(encoding="utf-8"))
    trace_p = rd / "ast5_trace.json"
    trace = json.loads(trace_p.read_text(encoding="utf-8")) if trace_p.exists() else {"families": []}
    rows = trace.get("families", [])

    def nodes(name):
        v = res.get(name)
        return (v.get("size") or {}).get("nodes") if v else None

    def passes(name):
        v = res.get(name)
        return bool(v and v["check"]["snap_pass"])

    out = {"task": spec.task, "variant": rd.parent.parent.name, "seed": rd.name, "learned": bool(log.get("learned")),
           "family": family, **formula_found(rows, family),
           "ast4_nodes": nodes("ast4"), "ast4_pass": passes("ast4"), "ast5_nodes": nodes("ast5"),
           "ast5_pass": passes("ast5"), "ast5_route": trace.get("winner"), "formula_py_pass": passes("formula"),
           **near_fields(trace.get("near"), family), "explain": (rd / "explain.md").exists()}
    if spec.output == "regression":
        X, _ = build_input_set(mlp, spec, 0)
        yn = net_output(mlp, spec, X)[0]
        with np.errstate(all="ignore"):
            yt = task.target(X).astype(float)
        ok = np.isfinite(yt)
        rng_n = float(yn.max() - yn.min()) or 1.0
        out["true_vs_net"] = round(float(np.abs(yn[ok] - yt[ok]).max()) / rng_n, 4)
    else:
        out["true_vs_net"] = None
    pysr_src = (rd / "pysr_irr_snap.py").read_text(encoding="utf-8") if (rd / "pysr_irr_snap.py").exists() else None
    out["pysr_found"] = passes("pysr_irr_snap") and pysr_found(pysr_src, family)

    model: dict = {}
    if len(spec.inputs) == 1:
        i = spec.inputs[0]
        model = kinks_1d(mlp, i.lo, i.hi)
        pts = model.get("kinks") or model.get("centres") or []
        model["optimal_knots"] = optimal_knots(lambda x: task.target(x[:, None]), i.lo, i.hi, len(pts))
        model["knot_gap"] = knot_gap(pts, model["optimal_knots"], i.hi - i.lo)
        x = np.linspace(i.lo, i.hi, N_GRID)
        with np.errstate(all="ignore"):
            e = np.abs(net_output(mlp, spec, x[:, None])[0] - task.target(x[:, None]))
        xm = float(x[int(np.nanargmax(e))])
        model["max_err_at"] = "edge" if min(xm - i.lo, i.hi - xm) <= EDGE_FRAC * (i.hi - i.lo) else "interior"
        out["hinge_only"] = hinge_fit(spec, mlp) if spec.output == "regression" else None
    elif spec.output == "binary":
        model = _model_2d_binary(spec.task, spec, mlp, rows)
        out["hinge_only"] = None
    else:
        model = _model_2d_regression(spec, mlp, task.target)
        out["hinge_only"] = None
    out["model"] = model
    hinge_nodes = (out["hinge_only"] or {}).get("nodes")
    out["won_on_size"] = won_on_size(out["formula_nodes"], hinge_nodes, out["ast4_nodes"] if out["ast4_pass"] else None)
    (rd / ANALYSIS).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    record_git(rd, "irr_analysis")
    return out
