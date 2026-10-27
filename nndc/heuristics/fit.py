"""P8: black-box candidate fit: sample the net and fit y ~ a*f + b over a small feature library.

HEURISTICS: median3/seed0-1. Approximate.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

from ..spec import DISCRETE_TYPES, Spec
from .cleanup import join_terms
from .snap import nice

MAX_INPUTS = 4
N_SAMPLES = 20_000
MIN_R2 = 0.999


def _features(names: list[str], X: np.ndarray) -> list[tuple[str, np.ndarray]]:
    col = {n: X[:, i] for i, n in enumerate(names)}
    feats = [(n, col[n]) for n in names]
    if len(names) >= 2:
        feats += [(f"sum({', '.join(names)})", X.sum(axis=1)),
                  (f"min({', '.join(names)})", X.min(axis=1)), (f"max({', '.join(names)})", X.max(axis=1))]
    for a, b in itertools.combinations(names, 2):
        feats += [(f"min({a}, {b})", np.minimum(col[a], col[b])), (f"max({a}, {b})", np.maximum(col[a], col[b])),
                  (f"abs({a} - {b})", np.abs(col[a] - col[b])),
                  (f"relu({a} - {b})", np.maximum(col[a] - col[b], 0)), (f"relu({b} - {a})", np.maximum(col[b] - col[a], 0))]
    if len(names) == 3:
        a, b, c = names
        feats.append((f"max(min({a}, {b}), min(max({a}, {b}), {c}))", np.median(X, axis=1)))
    return feats


def fit_expr(predict: Callable[[np.ndarray], np.ndarray], spec: Spec, seed: int = 0) -> str | None:
    if spec.output != "regression" or len(spec.inputs) > MAX_INPUTS:
        return None
    rng = np.random.default_rng(seed)
    cols = []
    for i in spec.inputs:
        if i.type in DISCRETE_TYPES:
            cols.append(rng.integers(int(i.lo), int(i.hi) + 1, N_SAMPLES).astype(np.float64))
        else:
            cols.append(rng.uniform(i.lo, i.hi, N_SAMPLES))
    X = np.stack(cols, axis=1)
    y = np.asarray(predict(X), dtype=np.float64)
    var = y.var()
    if var == 0:
        return None
    best = None
    for expr, feat in _features(spec.arg_names, X):
        A = np.stack([feat, np.ones_like(feat)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((A @ np.array([a, b]) - y) ** 2).mean() / var
        if r2 > MIN_R2 and (best is None or r2 > best[0]):
            best = (r2, expr, a, b)
    if best is None:
        return None
    _, expr, a, b = best
    a = nice(a, 0.05 * abs(a))
    b = nice(b, 0.01 * float(y.max() - y.min()))
    return join_terms([(a, expr), (b, None)])
