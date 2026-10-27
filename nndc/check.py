"""Compare any program with the original net: verdict, verification label, counterexamples.

Only stdlib + NumPy + sibling modules: this module is copied into blind folders.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from .lang import ProgramError, evaluate, parse_program
from .model import MLP
from .spec import DISCRETE_TYPES, Spec

EXHAUSTIVE_CAP = 10_000_000
N_RANDOM = 200_000
GRID_PER_DIM = 201
N_SEGMENTS = 2000
SEGMENT_SAMPLES = 64
BISECT_STEPS = 40
MAX_BOUNDARY = 200_000
MAX_COUNTEREXAMPLES = 20


@dataclass
class CheckResult:
    exact_pass: bool
    snap_pass: bool
    label: Literal["exhaustive", "empirical"]
    n_inputs: int
    max_abs_err: float = math.nan
    disagreements_outside: int = 0
    disagreements_inside: int = 0
    zone_fraction: float = 0.0
    margin: float = 0.0
    counterexamples: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None if math.isnan(v) else ("inf" if v > 0 else "-inf")
        return d


def _bounds(spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.array([i.lo for i in spec.inputs], dtype=np.float64)
    hi = np.array([i.hi for i in spec.inputs], dtype=np.float64)
    disc = np.array([i.type in DISCRETE_TYPES for i in spec.inputs])
    return lo, hi, disc


def _finish(X: np.ndarray, lo: np.ndarray, hi: np.ndarray, disc: np.ndarray) -> np.ndarray:
    X = np.clip(X, lo, hi)
    X[:, disc] = np.round(X[:, disc])
    return X


def _boundary_units(mlp: MLP, spec: Spec, X: np.ndarray) -> np.ndarray:
    """Columns whose sign changes mark activation or decision boundaries."""
    pre = mlp.preactivations(X)
    cols = [z.astype(np.float64) for z in pre[:-1]]
    out = pre[-1].astype(np.float64)
    if spec.output == "binary":
        cols.append(out[:, :1] - spec.threshold)
    elif spec.output == "argmax" and out.shape[1] <= 6:
        cols += [out[:, [a]] - out[:, [b]] for a, b in itertools.combinations(range(out.shape[1]), 2)]
    return np.concatenate(cols, axis=1) if cols else np.zeros((X.shape[0], 0))


def _boundary_points(mlp: MLP, spec: Spec, rng: np.random.Generator,
                     lo: np.ndarray, hi: np.ndarray, disc: np.ndarray) -> np.ndarray:
    d = lo.size
    A = rng.uniform(lo, hi, (N_SEGMENTS, d))
    B = rng.uniform(lo, hi, (N_SEGMENTS, d))
    t = np.linspace(0.0, 1.0, SEGMENT_SAMPLES)
    P = A[:, None, :] + t[None, :, None] * (B - A)[:, None, :]
    U = _boundary_units(mlp, spec, P.reshape(-1, d)).reshape(N_SEGMENTS, SEGMENT_SAMPLES, -1)
    if U.shape[2] == 0:
        return np.zeros((0, d))
    s = U > 0
    seg, k, unit = np.nonzero(s[:, :-1, :] != s[:, 1:, :])
    if seg.size == 0:
        return np.zeros((0, d))
    if seg.size > MAX_BOUNDARY // 4:
        keep = rng.choice(seg.size, MAX_BOUNDARY // 4, replace=False)
        seg, k, unit = seg[keep], k[keep], unit[keep]
    t0, t1 = t[k], t[k + 1]
    sign0 = s[seg, k, unit]
    Aseg, Dseg = A[seg], (B - A)[seg]
    rows = np.arange(seg.size)
    for _ in range(BISECT_STEPS):
        tm = (t0 + t1) / 2
        um = _boundary_units(mlp, spec, Aseg + tm[:, None] * Dseg)[rows, unit] > 0
        same = um == sign0
        t0 = np.where(same, tm, t0)
        t1 = np.where(same, t1, tm)
    mid = Aseg + ((t0 + t1) / 2)[:, None] * Dseg
    norm = np.linalg.norm(Dseg, axis=1, keepdims=True)
    direction = Dseg / np.where(norm == 0, 1, norm)
    delta = 1e-4 * np.linalg.norm(hi - lo)
    m32 = mid.astype(np.float32)
    pts = [mid, mid + delta * direction, mid - delta * direction,
           np.nextafter(m32, np.float32(np.inf)).astype(np.float64),
           np.nextafter(m32, np.float32(-np.inf)).astype(np.float64)]
    out = _finish(np.concatenate(pts), lo, hi, disc)
    if out.shape[0] > MAX_BOUNDARY:
        out = out[rng.choice(out.shape[0], MAX_BOUNDARY, replace=False)]
    return out


def build_input_set(mlp: MLP, spec: Spec, seed: int = 0,
                    extra: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    lo, hi, disc = _bounds(spec)
    d = lo.size
    if spec.is_discrete() and spec.domain_size() <= EXHAUSTIVE_CAP:
        axes = [np.arange(int(a), int(b) + 1, dtype=np.float64) for a, b in zip(lo, hi)]
        X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
        label = "exhaustive"
    else:
        rng = np.random.default_rng(seed)
        parts = [_finish(rng.uniform(lo, hi + disc, (N_RANDOM, d)) - 0.5 * disc, lo, hi, disc)]
        if not disc.any() and d <= 2:
            axes = [np.linspace(a, b, GRID_PER_DIM) for a, b in zip(lo, hi)]
            parts.append(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d))
        if d <= 12:
            corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=np.float64)
        else:
            corners = np.where(rng.integers(0, 2, (4096, d)).astype(bool), hi, lo)
        parts.append(corners)
        parts.append(_boundary_points(mlp, spec, rng, lo, hi, disc))
        X = np.concatenate(parts)
        label = "empirical"
    if extra is not None and len(extra):
        X = np.concatenate([X, np.asarray(extra, dtype=np.float64).reshape(-1, d)])
    return X, label


def net_output(mlp: MLP, spec: Spec, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = mlp.forward(X.astype(np.float32)).astype(np.float64)
    if spec.output == "regression":
        return out[:, 0], out[:, 0]
    if spec.output == "binary":
        return (out[:, 0] > spec.threshold).astype(np.float64), out[:, 0] - spec.threshold
    top2 = np.sort(out, axis=1)[:, -2:]
    return np.argmax(out, axis=1).astype(np.float64), top2[:, 1] - top2[:, 0]


def _rounding_bound(mlp: MLP, X: np.ndarray) -> np.ndarray:
    """Per-row bound on the float32 forward pass's rounding error, largest over the outputs.

    Propagates e_l = |W_l| e_{l-1} + eps*(fan_in+1) * (|W_l| |h_{l-1}| + |b_l|), starting from the
    rounding of the inputs to float32; ReLU does not increase the error.
    """
    eps = float(np.finfo(np.float32).eps)
    h = X.astype(np.float32).astype(np.float64)
    e = eps * np.abs(X)
    for i, (W, b) in enumerate(zip(mlp.weights, mlp.biases)):
        Wa = np.abs(W.astype(np.float64))
        e = e @ Wa.T + eps * (W.shape[1] + 1) * (np.abs(h) @ Wa.T + np.abs(b.astype(np.float64)))
        z = h @ W.T.astype(np.float64) + b
        h = np.maximum(z, 0) if i < len(mlp.weights) - 1 else z
    return e.max(axis=1)


def program_decision(spec: Spec, raw: np.ndarray) -> np.ndarray:
    if spec.output == "binary":
        return (raw > spec.threshold).astype(np.float64)
    return raw


def _inputs_dict(spec: Spec, x: np.ndarray) -> dict:
    return {i.name: (int(v) if i.type in DISCRETE_TYPES else float(v)) for i, v in zip(spec.inputs, x)}


def check_program(src: str, mlp: MLP, spec: Spec, seed: int = 0,
                  extra_inputs: np.ndarray | None = None) -> CheckResult:
    X, label = build_input_set(mlp, spec, seed, extra_inputs)
    try:
        raw = evaluate(parse_program(src, spec.arg_names), X)
    except ProgramError as e:
        return CheckResult(False, False, label, X.shape[0], error=str(e))
    n_dec, n_score = net_output(mlp, spec, X)
    finite = np.isfinite(raw)
    if spec.output == "regression":
        err = np.where(finite, np.abs(raw - n_dec), np.inf)
        exact_ok = err <= spec.atol + spec.rtol * np.abs(n_dec)
        tol = spec.snap_tol_frac * float(n_dec.max() - n_dec.min())
        snap_ok = (err <= tol) | exact_ok   # a constant output has range 0, hence tol 0
        bad_out, bad_in = ~snap_ok, snap_ok & ~exact_ok
        severity = err
        r = CheckResult(bool(exact_ok.all()), bool(snap_ok.all()), label, X.shape[0],
                        max_abs_err=float(err.max()), disagreements_outside=int(bad_out.sum()),
                        margin=tol)
    else:
        p_dec = program_decision(spec, raw)
        mismatch = (p_dec != n_dec) | ~finite
        score = np.abs(n_score)
        m = spec.margin_frac * float(np.percentile(score, 95))
        in_zone = score < m
        bad_out, bad_in = mismatch & ~in_zone, mismatch & in_zone
        severity = score
        # within the net's own float32 rounding of the decision line the class is a tie, not a disagreement
        tie = score <= spec.atol + 2 * _rounding_bound(mlp, X)
        # the zone's share of the input domain: uniform part of the input set only (boundary points cluster)
        domain = in_zone if label == "exhaustive" else in_zone[:N_RANDOM]
        r = CheckResult(not (mismatch & ~tie).any(), not bad_out.any(), label, X.shape[0],
                        disagreements_outside=int(bad_out.sum()), disagreements_inside=int(bad_in.sum()),
                        zone_fraction=float(domain.mean()), margin=m)
    idx = list(np.nonzero(bad_out)[0][np.argsort(-severity[bad_out], kind="stable")])
    idx += list(np.nonzero(bad_in)[0][np.argsort(-severity[bad_in], kind="stable")])
    for i in idx[:MAX_COUNTEREXAMPLES]:
        r.counterexamples.append({"inputs": _inputs_dict(spec, X[i]), "net": float(n_dec[i]),
                                  "program": float(raw[i]) if finite[i] else str(raw[i]),
                                  "in_zone": bool(bad_in[i])})
    return r


def format_result(r: CheckResult, spec: Spec, max_examples: int = 10) -> str:
    if r.error:
        return f"ERROR: {r.error}\n"
    head = "PASS (exact)" if r.exact_pass else "PASS (snap)" if r.snap_pass else "FAIL"
    lines = [f"{head}  [{r.label}, {r.n_inputs} inputs]"]
    if spec.output == "regression":
        lines.append(f"max |program - net| = {r.max_abs_err:.6g}; snap tolerance = {r.margin:.6g}; "
                     f"inputs beyond snap tolerance: {r.disagreements_outside}")
    else:
        lines.append(f"disagreements with the net: {r.disagreements_outside} where the net is confident, "
                     f"{r.disagreements_inside} where it is unsure (unsure zone: score < {r.margin:.4g}, "
                     f"{100 * r.zone_fraction:.2f}% of inputs)")
    if r.snap_pass and not r.exact_pass:
        lines.append("differs from the net only within the allowed snap tolerance / unsure zone")
    for c in r.counterexamples[:max_examples]:
        xs = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in c["inputs"].items())
        where = "unsure zone" if c["in_zone"] else "OUTSIDE tolerance" if spec.output == "regression" else "net confident"
        lines.append(f"  {xs} -> net {c['net']:.6g}, program {c['program']}  ({where})")
    return "\n".join(lines) + "\n"
