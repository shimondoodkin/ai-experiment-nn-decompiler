"""`nndc explain`: per run, the formula (the passing family or conic program, else the near formula with a description
of where it is off) and the gate view of the net's units, written to explain.md.

Design: docs/design/2026-10-31-formula-residual-gate-view-design.md. Blind: it reads the weights, the blind spec and
ast5_trace.json only. NumPy only.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .gate_view import groups, tolerance, unit_table
from .lang import evaluate, parse_program
from .model import MLP, load_npz
from .spec import Spec

N_GRID_1D = 2001
N_SAMPLE_2D = 100_000
EDGE = 0.05          # within this share of the box width of a face: "edge"
N_SLICES = 10
N_RINGS = 10
N_SECTORS = 12
FORMULA_ROUTES = ("family", "conic")


def _sample(spec: Spec) -> np.ndarray:
    if len(spec.inputs) == 1:
        i = spec.inputs[0]
        return np.linspace(i.lo, i.hi, N_GRID_1D)[:, None]
    rng = np.random.default_rng(0)
    return np.stack([rng.uniform(i.lo, i.hi, N_SAMPLE_2D) for i in spec.inputs], axis=1)


def _box_centre(spec: Spec) -> tuple[float, ...]:
    return tuple(float((i.lo + i.hi) / 2) for i in spec.inputs)


def _inner(src: str) -> str:
    """A program returning the formula's inner expression: the derived variable u if there is one, else the argument
    of the first exp/log/sqrt, else the denominator of the first division, else the returned expression."""
    fn = ast.parse(src).body[0]
    assigns = [s for s in fn.body if isinstance(s, ast.Assign)]
    ret = next(s for s in fn.body if isinstance(s, ast.Return)).value
    names = {t.id for a in assigns for t in a.targets if isinstance(t, ast.Name)}
    inner = None
    if "u" in names:
        inner = ast.Name("u")
    if inner is None:
        inner = next((n.args[0] for n in ast.walk(ret) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id in ("exp", "log", "sqrt") and n.args), None)
    if inner is None:
        inner = next((n.right for n in ast.walk(ret) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)), ret)
    args = ", ".join(a.arg for a in fn.args.args)
    body = "".join(f"    {ast.unparse(a)}\n" for a in assigns)
    return f"def net({args}):\n{body}    return {ast.unparse(inner)}\n"


def formula_centre(src: str, spec: Spec) -> tuple[float, float]:
    """The vertex of the formula's inner quadratic when its quadratic part is definite, else the box centre."""
    box = _box_centre(spec)
    try:
        prog = parse_program(_inner(src), spec.arg_names)
    except (SyntaxError, ValueError, StopIteration, IndexError):
        return box
    rng = np.random.default_rng(1)
    X = np.stack([rng.uniform(i.lo, i.hi, 2000) for i in spec.inputs], axis=1)
    with np.errstate(all="ignore"):
        q = np.asarray(evaluate(prog, X), dtype=np.float64)
    if not np.isfinite(q).all():
        return box
    x0, x1 = X[:, 0], X[:, 1]
    A = np.stack([x0 * x0, x1 * x1, x0 * x1, x0, x1, np.ones_like(x0)], axis=1)
    p = np.linalg.lstsq(A, q, rcond=None)[0]
    if np.abs(A @ p - q).max() > 1e-6 * (1 + np.abs(q).max()):
        return box                                   # not a quadratic
    a, b, c, d, e = p[:5]
    H = np.array([[2 * a, c], [c, 2 * b]])
    if np.linalg.det(H) <= 1e-9 * (1 + np.abs(H).max() ** 2):
        return box                                   # saddle, or degenerate (a pure line)
    cx, cy = np.linalg.solve(H, [-d, -e])
    return round(float(cx), 9) + 0.0, round(float(cy), 9) + 0.0     # + 0.0: no -0.0


def _binned(r: np.ndarray, idx: np.ndarray, n: int) -> tuple[list, list]:
    mean, worst = [], []
    for k in range(n):
        s = r[idx == k]
        mean.append(float(s.mean()) if len(s) else None)
        worst.append(float(np.abs(s).max()) if len(s) else None)
    return mean, worst


def describe_residual(spec: Spec, predict: Callable[[np.ndarray], np.ndarray], src: str) -> dict:
    """Where the program is off from the net, in shares of the net's output range: the max error and whether it is
    at the box edge; 1 input: the mean (signed) and max error in 10 equal slices; 2 inputs: by distance from the
    formula's centre in 10 rings and by angle in 12 sectors."""
    X = _sample(spec)
    y = np.asarray(predict(X), dtype=np.float64)
    with np.errstate(all="ignore"):
        f = np.asarray(evaluate(parse_program(src, spec.arg_names), X), dtype=np.float64)
    R = float(y.max() - y.min()) or 1.0
    r = (f - y) / R
    a = np.where(np.isfinite(r), np.abs(r), np.inf)
    i = int(np.argmax(a))
    lo = np.array([s.lo for s in spec.inputs])
    hi = np.array([s.hi for s in spec.inputs])
    near_face = np.minimum(X[i] - lo, hi - X[i]) <= EDGE * (hi - lo)
    out = {"max_err": float(a[i]), "where": "edge" if near_face.any() else "inside",
           "at": [float(v) for v in X[i]]}
    r = np.where(np.isfinite(r), r, 0.0)
    if len(spec.inputs) == 1:
        idx = np.clip(((X[:, 0] - lo[0]) / (hi[0] - lo[0]) * N_SLICES).astype(int), 0, N_SLICES - 1)
        out["by_slice"], out["by_slice_max"] = _binned(r, idx, N_SLICES)
        return out
    c = formula_centre(src, spec)
    dx, dy = X[:, 0] - c[0], X[:, 1] - c[1]
    d = np.hypot(dx, dy)
    ring = np.clip((d / d.max() * N_RINGS).astype(int), 0, N_RINGS - 1)
    sector = np.clip((np.degrees(np.arctan2(dy, dx)) % 360 / (360 / N_SECTORS)).astype(int), 0, N_SECTORS - 1)
    out["centre"] = c
    out["ring_width"] = float(d.max() / N_RINGS)
    out["by_ring"], out["by_ring_max"] = _binned(r, ring, N_RINGS)
    out["by_sector"], out["by_sector_max"] = _binned(r, sector, N_SECTORS)
    return out


def _pct(xs: list) -> str:
    return " ".join("-" if x is None else f"{100 * x:+.1f}" for x in xs)


def _code(src: str) -> list[str]:
    return ["```python", src.rstrip("\n"), "```"]


def _formula_section(spec: Spec, mlp: MLP, trace: dict | None) -> list[str]:
    out = ["## Formula", ""]
    if trace is None:
        return out + ["There is no ast5 trace for this run (run `nndc ast --name ast5` first)."]
    passing = [c for c in trace.get("candidates", []) if c.get("route") in FORMULA_ROUTES
               and c.get("check", {}).get("snap_pass")]
    if passing:
        best = min(passing, key=lambda c: c["nodes"])
        return out + [f"A {best['route']} program passes the check ({best['nodes']} nodes):", ""] + _code(best["src"])
    near = trace.get("near")
    if not near:
        return out + ["No formula program: no family fitted the net and no near formula was recorded."]
    if near["family"] == "conic":
        out += [f"No formula passes the check. Near formula (conic): it decides {100 * near['err']:.1f}% of the "
                f"checker's inputs differently where the net is confident.", ""]
        return out + _code(near["src"])
    out += [f"No formula passes the check. Near formula ({near['family']}, {near['terms']} correction terms, "
            f"{near['nodes']} nodes), max error {100 * near['err']:.1f}% of the output range (the check allows "
            f"{100 * spec.snap_tol_frac:.0f}%):", ""] + _code(near["src"])
    predict = lambda X: mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)   # noqa: E731
    try:
        res = describe_residual(spec, predict, near["src"])
    except (SyntaxError, ValueError) as e:
        return out + ["", f"(residual not described: {e})"]
    at = ", ".join(f"{n} = {v:.3g}" for n, v in zip(spec.arg_names, res["at"]))
    out += ["", "Where it is off (formula minus net, % of the output range):",
            f"- largest error {100 * res['max_err']:.1f}%, {'at the box edge' if res['where'] == 'edge' else 'inside the box'} ({at})"]
    if "by_slice" in res:
        out += [f"- mean error in {N_SLICES} equal slices of {spec.arg_names[0]}: {_pct(res['by_slice'])}",
                f"- largest error per slice: {_pct(res['by_slice_max'])}"]
    else:
        cx, cy = res["centre"]
        out += [f"- by distance from ({cx:.3g}, {cy:.3g}), rings {res['ring_width']:.3g} wide, mean: "
                f"{_pct(res['by_ring'])}; largest: {_pct(res['by_ring_max'])}",
                f"- by angle, {N_SECTORS} sectors of {360 // N_SECTORS} degrees from the x0 axis, mean: "
                f"{_pct(res['by_sector'])}; largest: {_pct(res['by_sector_max'])}"]
    return out


def _units_table(rows: list[dict]) -> list[str]:
    zones = list(rows[0]["zones"]) if rows else []
    head = ["unit", "angle", "offset", "width", *zones, "magnitude", "sign", "stand-in"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        stand = r["stand_in"] or f"none (best {r['stand_in_err']:.3g})"
        cells = [str(r["unit"]), f"{r['angle']:.0f}", f"{r['offset']:.3g}", f"{r['width']:.3g}",
                 *(f"{r['zones'][z]:.2f}" for z in zones), f"{r['magnitude']:.3g}", "+" if r["sign"] >= 0 else "-",
                 stand]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _group_line(g: dict, rows: dict[int, dict], tol: float) -> str:
    ids = ", ".join(str(u) for u in g["units"])
    s = g["summary"]
    if g["kind"] == "negligible":
        return f"- negligible: units {ids} (magnitude {s['magnitude']:.3g} together; a quarter of the tolerance is {tol / 4:.3g})"
    if g["kind"] == "lines":
        return f"- lines: units {ids} (each a constant or a line on the box; together one linear term)"
    if g["kind"] == "ring":
        width = float(np.median([rows[u]["width"] for u in g["units"]]))
        sign = "positive" if s["sign"] > 0 else "negative"
        return (f"- ring: {s['count']} units ({ids}) with gates at distance {s['offset_min']:.2f}-{s['offset_max']:.2f}, "
                f"switch width ~{width:.2g}, magnitude ~{s['magnitude_median']:.2g} each, {sign}")
    if g["kind"] == "same line":
        return f"- same line: units {ids} (gate angle {s['angle']:.0f} degrees, offset {s['offset']:.3g})"
    return f"- the rest: units {ids}"


def explain(spec: Spec, mlp: MLP, trace: dict | None) -> str:
    out = ["# Explain", "", f"Inputs: {', '.join(f'{i.name} in [{i.lo:g}, {i.hi:g}]' for i in spec.inputs)}; "
           f"output: {spec.output}.", ""]
    out += _formula_section(spec, mlp, trace) + ["", "## Units", ""]
    if len(mlp.weights) != 2:
        return "\n".join(out + [f"The gate view needs one hidden layer; this net has {len(mlp.weights) - 1}.", ""])
    try:
        rows = unit_table(spec, mlp)
    except ValueError as e:
        return "\n".join(out + [f"No gate view: {e}.", ""])
    tol = tolerance(spec, mlp)
    act = "relu" if mlp.acts[0] == "relu" else "sigmoid"
    out += [f"Each {act} unit as a gate: the gate line's angle (degrees) and signed offset from the origin, the switch "
            f"width (4/|w|), the share of the box in each zone, the magnitude (range of the unit's term over the box), "
            f"the sign of its output weight, and the simplest stand-in within a quarter of the tolerance "
            f"({tol:.3g}, 2% of the output range).", ""] + _units_table(rows)
    width = max(i.hi - i.lo for i in spec.inputs)
    by_id = {r["unit"]: r for r in rows}
    out += ["", "## Groups", ""] + [_group_line(g, by_id, tol) for g in groups(rows, tol, width)]
    return "\n".join(out + [""])


def explain_run(rd: Path) -> str:
    """Write rd/explain.md from the run's weights, blind spec and ast5_trace.json; record `explain` in run_git.txt."""
    from .run import record_git

    rd = Path(rd)
    spec = Spec.from_json(Spec.load(rd / "spec.json").blind())
    mlp = load_npz(rd / "weights.npz")
    tp = rd / "ast5_trace.json"
    trace = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else None
    text = explain(spec, mlp, trace)
    (rd / "explain.md").write_text(text, encoding="utf-8")
    record_git(rd, "explain")
    return text
