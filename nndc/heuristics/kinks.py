"""Kink merge for one-layer sigmoid nets.

Each sigmoid becomes PIECES straight pieces (chords on knots uniform in its pre-activation over the input box),
so the net becomes an affine part plus ReLU kinks. Kinks on the same line add up, and many nearly cancel; a few
strong lines, refined and refitted to the net, give a short program. It counts only if it passes the checker.

Design: docs/design/2026-10-29-kink-merge-design.md (parameters fixed there before any tool code).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..check import check_program
from ..lang import parse_program, program_size
from ..model import MLP, activate
from ..spec import Spec
from ..symnet import Affine
from .cleanup import program_from_expr
from .decision_list import nicest_in
from .motifs import ReluSum, motif_expr
from .sigmoid import SNAP_TOLS, minimax
from .snap import nice

PIECES = 17
COS_MIN = 0.995                 # normals at least this parallel lie on one line ...
MERGE_OFFS = (0.02, 0.1)        # ... when their offsets (unit box) differ by less than this
MAX_LINES = 8
OFF_RANGE = 0.1                 # offset search: +-OFF_RANGE in OFF_STEPS steps, OFF_ROUNDS rounds
OFF_STEPS = 21
OFF_ROUNDS = 3
FIT_SAMPLES = 20_000
FIT_SEED = 0
SNAP_LEVELS = (*SNAP_TOLS, 0.0)  # coarsest first; 0 = fold and refit, no rounding (ast3; ast2 tried the raw fit last)
SEARCH_SAMPLES = 2000          # the offset search fits on this many of the samples (runtime; set before the freeze)
CANCEL_FRAC = 0.25              # a merged line nearly cancels when |strength| < CANCEL_FRAC * sum |strengths|


@dataclass
class Kink:
    normal: np.ndarray          # unit-box coordinates, |normal| = 1, first nonzero component positive
    offset: float
    strength: float             # term: strength * max(normal @ u + offset, 0)


@dataclass
class Expansion:
    const: float
    linear: np.ndarray
    kinks: list[Kink]


@dataclass
class RealFit:
    const: float
    linear: dict[str, float]
    terms: list[tuple[dict[str, float], float, float]]   # (normal over input names, t, strength):
                                                         # strength * max(normal . x - t, 0); max |normal| = 1


@dataclass
class Line:
    normal: np.ndarray
    offset: float
    strength: float
    gross: float                # sum of the merged kinks' |strength|
    count: int


def applies(spec: Spec, mlp: MLP) -> str | None:
    """None when the route applies to this net, else the reason it does not."""
    if spec.output == "argmax":
        return "argmax output"
    if len(mlp.weights) != 2:
        return "not one hidden layer"
    if mlp.acts != ["sigmoid"]:
        return "not all sigmoid"
    return None


def unit_box(spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([float(i.lo) for i in spec.inputs])
    return lo, np.array([float(i.hi) for i in spec.inputs]) - lo


def _corners(n: int) -> np.ndarray:
    return np.array(list(itertools.product([0.0, 1.0], repeat=n)))


def expand(spec: Spec, mlp: MLP, pieces: int = PIECES) -> Expansion:
    lo, span = unit_box(spec)
    W, b = mlp.weights[0].astype(np.float64), mlp.biases[0].astype(np.float64)
    v, c = mlp.weights[1][0].astype(np.float64), float(mlp.biases[1][0])
    Wu, bu = W * span, W @ lo + b                       # z = Wu @ u + bu
    corners = _corners(len(lo))
    const, linear, kinks = c, np.zeros(len(lo)), []
    for j in range(len(v)):
        zc = corners @ Wu[j] + bu[j]
        if zc.max() - zc.min() <= 1e-12:                # constant pre-activation: a constant unit
            const += v[j] * float(activate("sigmoid", np.array([bu[j]]))[0])
            continue
        t = np.linspace(zc.min(), zc.max(), pieces + 1)
        s = np.diff(activate("sigmoid", t)) / np.diff(t)
        # chord interpolation: sig(t0) + s0 (z - t0) + sum_i (s_i - s_{i-1}) max(z - t_i, 0)
        const += v[j] * (float(activate("sigmoid", t[:1])[0]) + s[0] * (bu[j] - t[0]))
        linear = linear + v[j] * s[0] * Wu[j]
        n = float(np.linalg.norm(Wu[j]))
        for i in range(1, pieces):
            normal, offset, st = Wu[j] / n, (bu[j] - t[i]) / n, v[j] * (s[i] - s[i - 1]) * n
            if normal[np.flatnonzero(np.abs(normal) > 1e-12)[0]] < 0:     # st max(y,0) = st max(-y,0) + st y
                const += st * offset
                linear = linear + st * normal
                normal, offset = -normal, -offset
            kinks.append(Kink(normal, float(offset), float(st)))
    return Expansion(float(const), linear, kinks)


def evaluate_expansion(e: Expansion, U: np.ndarray) -> np.ndarray:
    y = e.const + U @ e.linear
    for k in e.kinks:
        y = y + k.strength * np.maximum(U @ k.normal + k.offset, 0.0)
    return y


def merge_lines(kinks: list[Kink], off: float, cos_min: float = COS_MIN) -> list[Line]:
    """Kinks on one line (greedy first match) add up; normal and offset are |strength|-weighted means.
    An antiparallel kink on the same line joins as (-n, -d, strength): s max(y, 0) = s max(-y, 0) + s y, and the
    affine s y is left to refine(), which refits the affine part."""
    groups: list[tuple[np.ndarray, float, list[Kink]]] = []
    for k in kinks:
        for g in groups:
            c = float(g[0] @ k.normal)
            if abs(c) > cos_min:
                kk = k if c > 0 else Kink(-k.normal, -k.offset, k.strength)
                if abs(g[1] - kk.offset) < off:
                    g[2].append(kk)
                    break
        else:
            groups.append((k.normal, k.offset, [k]))
    lines = []
    for _, _, ks in groups:
        w = np.array([abs(k.strength) for k in ks]) + 1e-300
        normal = sum(wi * k.normal for wi, k in zip(w, ks))
        normal = normal / np.linalg.norm(normal)
        offset = float(sum(wi * k.offset for wi, k in zip(w, ks)) / w.sum())
        lines.append(Line(normal, offset, float(sum(k.strength for k in ks)),
                          float(sum(abs(k.strength) for k in ks)), len(ks)))
    return sorted(lines, key=lambda ln: -abs(ln.strength))


def nearly_cancelled(line: Line) -> bool:
    return line.count >= 2 and abs(line.strength) < CANCEL_FRAC * line.gross


def _design(lines: list[Line], offsets: list[float], U: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(U)), U] + [np.maximum(U @ ln.normal + d, 0.0) for ln, d in zip(lines, offsets)])


def refine(lines: list[Line], U: np.ndarray, y: np.ndarray) -> tuple[list[Line], np.ndarray, float]:
    """Coordinate search on the offsets (on the first SEARCH_SAMPLES rows); coefficients from the minimax LP on
    all rows. Returns (lines, [c, a.., s..], max err)."""
    offsets = [ln.offset for ln in lines]
    Us, ys = U[:SEARCH_SAMPLES], y[:SEARCH_SAMPLES]
    _, err = minimax(_design(lines, offsets, Us), ys)
    for _ in range(OFF_ROUNDS):
        for i in range(len(lines)):
            center = offsets[i]
            for d in center + np.linspace(-OFF_RANGE, OFF_RANGE, OFF_STEPS):
                trial = offsets[:i] + [float(d)] + offsets[i + 1:]
                _, e2 = minimax(_design(lines, trial, Us), ys)
                if e2 < err:
                    offsets, err = trial, e2
    coefs, err = minimax(_design(lines, offsets, U), y)
    k = 1 + U.shape[1]
    out = [Line(ln.normal, d, float(s), ln.gross, ln.count) for ln, d, s in zip(lines, offsets, coefs[k:])]
    return out, coefs, float(err)


def to_real(lines: list[Line], coefs: np.ndarray, spec: Spec) -> RealFit:
    """Unit-box fit -> real input coordinates; each kink normal scaled so its largest |component| is 1."""
    lo, span = unit_box(spec)
    names = spec.arg_names
    k = len(names)
    a_u = np.asarray(coefs[1:1 + k], dtype=np.float64)
    const = float(coefs[0] - np.sum(a_u * lo / span))
    linear = {names[i]: float(a_u[i] / span[i]) for i in range(k)}
    terms = []
    for ln, s in zip(lines, coefs[1 + k:]):
        r = ln.normal / span                              # line: r . x + off = 0
        off = ln.offset - float(np.sum(ln.normal * lo / span))
        a = float(np.abs(r).max())
        terms.append(({names[i]: float(r[i] / a) for i in range(k) if r[i] != 0}, -off / a, float(s) * a))
    return RealFit(const, linear, terms)


def _nicest_rel(s: float, tol: float) -> float:
    return 0.0 if s == 0 else nicest_in(s - tol * abs(s), s + tol * abs(s))


def _hinges(X: np.ndarray, names: list[str], terms) -> list[np.ndarray]:
    return [np.maximum(X @ np.array([n.get(v, 0.0) for v in names]) - t, 0.0) for n, t, *_ in terms]


def snap_fit(fit: RealFit, spec: Spec, tol: float, U: np.ndarray, y: np.ndarray, window: str = "line",
             round_const: bool = True) -> RealFit:
    """Direction, position (tol * the line's own range over the box; window="span": tol * the largest input range,
    as in ast2), LP refit, coefficients (relative tol), then the constant (midrange of the error, rounded within
    tol * output range unless round_const is False). A line that leaves the box is folded into the affine part.
    tol = 0 folds and refits but rounds nothing."""
    lo, span = unit_box(spec)
    names = spec.arg_names
    X = lo + U * span
    box = lo + _corners(len(names)) * span
    geo: dict[tuple, tuple[dict[str, float], float]] = {}
    for n, t, _ in fit.terms:
        big = max(n, key=lambda v: abs(n[v]))
        n2 = {v: (c if v == big else nice(c, tol)) for v, c in n.items()}
        n2 = {v: c for v, c in n2.items() if c != 0}
        width = (sum(abs(n2.get(v, 0.0)) * float(s) for v, s in zip(names, span)) if window == "line"
                 else float(np.max(span)))
        t2 = nicest_in(t - tol * width, t + tol * width)
        z = box @ np.array([n2.get(v, 0.0) for v in names]) - t2
        if z.min() >= 0 or z.max() <= 0:                 # always on (affine, refitted below) or always off
            continue
        geo[(tuple(sorted(n2.items())), t2)] = (n2, t2)  # identical snapped lines become one term
    lines = list(geo.values())
    H = _hinges(X, names, lines)
    c, _ = minimax(np.column_stack([np.ones(len(X)), X] + H), y)
    k = len(names)
    linear = {names[i]: _nicest_rel(float(c[1 + i]), tol) for i in range(k)}
    strengths = [_nicest_rel(float(s), tol) for s in c[1 + k:]]
    rest = y - X @ np.array([linear[v] for v in names]) - sum((s * h for s, h in zip(strengths, H)), np.zeros(len(X)))
    const = float((rest.max() + rest.min()) / 2)
    if tol > 0 and round_const:
        r = float(y.max() - y.min())
        const = nicest_in(const - tol * r, const + tol * r)
    return RealFit(const, linear, [(n, t, s) for (n, t), s in zip(lines, strengths)])


def fit_program(fit: RealFit, spec: Spec) -> str:
    """The fit as a program, written through the ReLU motif rewriting (abs / max pairs where they apply)."""
    terms = [(s, Affine({v: c for v, c in n.items() if c != 0}, -t)) for n, t, s in fit.terms if s != 0]
    rs = ReluSum(terms, Affine({v: c for v, c in fit.linear.items() if c != 0}, fit.const))
    return program_from_expr(motif_expr(rs), spec)


VARIANTS = (("line", True), ("line", False), ("span", True), ("span", False))   # ast4: (window, round_const)


def best_candidate(cands: list[dict]) -> dict:
    """ast4's pick among passing candidates: fewest nodes, then exact, rounded constant, line window, coarser
    level, merge offset 0.02 (docs/design/2026-10-30-ast4-design.md)."""
    return min(cands, key=lambda c: (c["nodes"], not c["exact"], not c["round_const"], c["window"] != "line",
                                     -c["tol"], c["off"]))


def kink_merge(spec: Spec, mlp: MLP, version: str = "ast3") -> tuple[str | None, dict]:
    """The route: smallest m first. ast3: merge offset 0.02 before 0.1, snap ladder coarsest first, the first
    program that passes the checker wins. ast4: every offset, level and snap variant for this m is checked, and the
    passing one with the fewest nodes wins (best_candidate); otherwise the next m."""
    trace: dict = {"applies": applies(spec, mlp), "per_offset": [], "winner": None}
    if trace["applies"] is not None:
        return None, trace
    lo, span = unit_box(spec)
    d = len(lo)
    rng = np.random.default_rng(FIT_SEED)
    U = np.vstack([rng.uniform(0.0, 1.0, (FIT_SAMPLES, d)), _corners(d)])
    y = mlp.forward((lo + U * span).astype(np.float32))[:, 0].astype(np.float64)
    kinks = expand(spec, mlp).kinks
    line_sets = {}
    for off in MERGE_OFFS:
        line_sets[off] = merge_lines(kinks, off)
        trace["per_offset"].append({"off": off, "kinks": len(kinks), "lines": len(line_sets[off]),
                                    "cancelled": sum(nearly_cancelled(ln) for ln in line_sets[off])})
    variants = VARIANTS if version == "ast4" else VARIANTS[:1]
    seen: set[str] = set()
    for m in range(1, MAX_LINES + 1):
        passing = []
        for off in MERGE_OFFS:
            if len(line_sets[off]) < m:
                continue
            lines, coefs, _ = refine(line_sets[off][:m], U, y)
            fit = to_real(lines, coefs, spec)
            for tol in SNAP_LEVELS:
                for window, round_const in variants:
                    f = snap_fit(fit, spec, tol, U, y, window, round_const)
                    src = fit_program(f, spec)
                    if src in seen:
                        continue
                    seen.add(src)
                    r = check_program(src, mlp, spec)
                    if not r.snap_pass:
                        continue
                    c = {"off": off, "m": m, "tol": tol, "window": window, "round_const": round_const,
                         "exact": bool(r.exact_pass), "src": src, "fit": f,
                         "nodes": program_size(parse_program(src, spec.arg_names))["nodes"]}
                    if version != "ast4":
                        return _won(c, trace)
                    passing.append(c)
        if passing:
            return _won(best_candidate(passing), trace)
    return None, trace


def _won(c: dict, trace: dict) -> tuple[str, dict]:
    trace["winner"] = {k: c[k] for k in ("off", "m", "tol", "window", "round_const")} | {
        "lines": [{"normal": n, "t": t, "strength": s} for n, t, s in c["fit"].terms]}
    return c["src"], trace
