"""F2 (ast5): a conic decision boundary for 2-input binary nets: q(x) < c or q(x) > c, q a general quadratic.

Design: docs/design/2026-10-30-irrational-functions-design.md. Boundary points by bisection between sample pairs that
the net decides differently, a least-squares conic through them, then sequential snapping. Approximate: the checker
decides.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..check import check_program, net_output
from ..lang import ProgramError, parse_program, program_size
from ..model import MLP
from ..spec import Spec
from .cleanup import join_terms, program_from_expr
from .families import QUAD_ATOMS, SNAP_LEVELS, _is_nice, _quad_cols
from .snap import nice

N_SAMPLE = 20_000
N_BOUNDARY = 2000
MIN_BOUNDARY = 20
BISECT = 30
SLACK = 0.001        # while snapping, the sample may disagree with the net on this much more than the plain fit


def _boundary(spec: Spec, decide: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(i.lo, i.hi, N_SAMPLE) for i in spec.inputs], axis=1)
    d = np.asarray(decide(X)).astype(int)
    Y = X[rng.permutation(N_SAMPLE)]
    dy = np.asarray(decide(Y)).astype(int)
    diff = np.nonzero(d != dy)[0][:N_BOUNDARY]
    A, B, da = X[diff], Y[diff], d[diff]
    for _ in range(BISECT):
        M = (A + B) / 2
        same = np.asarray(decide(M)).astype(int) == da
        A = np.where(same[:, None], M, A)
        B = np.where(same[:, None], B, M)
    return (A + B) / 2, X, d


def _design(P: np.ndarray) -> np.ndarray:
    """Columns x0², x1², x0·x1, x0, x1, -1: a coefficient vector w puts the boundary at D @ w = 0 (w[5] = c)."""
    cols = _quad_cols(P)
    cols[5] = -cols[5]
    return np.stack(cols, axis=1)


def _decisions(w: np.ndarray, less: bool, X: np.ndarray) -> np.ndarray:
    q = _design(X)[:, :5] @ w[:5]
    return (q < w[5] if less else q > w[5]).astype(int)


def _emit(w: np.ndarray, less: bool, spec: Spec) -> str:
    q = join_terms([(float(w[i]), QUAD_ATOMS[i]) for i in range(5)])
    return program_from_expr(f"{q} {'<' if less else '>'} {join_terms([(float(w[5]), None)])}", spec)


def _refit(w: np.ndarray, fixed: np.ndarray, D: np.ndarray) -> np.ndarray:
    free = np.nonzero(~fixed)[0]
    if not len(free):
        return w
    rhs = -(D[:, fixed] @ w[fixed])
    w = w.copy()
    w[free] = np.linalg.lstsq(D[:, free], rhs, rcond=None)[0]
    return w


def conic_candidate(spec: Spec, mlp: MLP | None, decide: Callable[[np.ndarray], np.ndarray] | None = None,
                    check: Callable[[str], bool] | None = None) -> tuple[str | None, dict]:
    """The conic program, or None; and an info row. decide and check are test hooks (default: the net's decision,
    and the checker's snap verdict against the net)."""
    info = {"family": "conic", "n_boundary": 0, "coefs": None, "snapped": None, "passed": False, "nodes": None,
            "src": None, "terms": 0}
    if spec.output != "binary" or len(spec.inputs) != 2:
        return None, info
    if decide is None:
        decide = lambda X: net_output(mlp, spec, X)[0]   # noqa: E731
    if check is None:
        check = lambda src: check_program(src, mlp, spec).snap_pass   # noqa: E731
    P, X, d = _boundary(spec, decide)
    info["n_boundary"] = int(len(P))
    if len(P) < MIN_BOUNDARY:
        return None, info
    D = _design(P)
    w = np.linalg.svd(D / np.abs(D).max(axis=0).clip(1e-12), full_matrices=False)[2][-1]
    w = w / np.abs(D).max(axis=0).clip(1e-12)            # undo the column scaling
    lead = int(np.argmax(np.abs(w[:3]))) if np.abs(w[:3]).max() > 1e-9 * np.abs(w).max() else int(np.argmax(np.abs(w)))
    w = w / w[lead]
    inside = _design(X)[:, :5] @ w[:5] < w[5]
    less = bool(d[inside].mean() >= 0.5) if inside.any() else False
    info["coefs"] = [float(x) for x in w]
    base = float((_decisions(w, less, X) != d).mean())
    limit = base + SLACK

    fixed = np.zeros(6, bool)
    fixed[lead] = True
    hopeless: set[int] = set()
    while (~fixed).any():
        best = None
        for i in np.nonzero(~fixed)[0]:
            if i in hopeless:
                continue
            for lev in SNAP_LEVELS:
                if best is not None and lev <= best[0]:
                    break
                c = nice(float(w[i]), lev * max(1.0, abs(float(w[i]))))
                if not _is_nice(c):
                    continue
                f2 = fixed.copy()
                f2[i] = True
                w2 = w.copy()
                w2[i] = c
                w2 = _refit(w2, f2, D)
                if float((_decisions(w2, less, X) != d).mean()) <= limit:
                    best = (lev, i, w2)
                    break
            else:
                if best is None:
                    hopeless.add(i)
        if best is None:
            break
        _, i, w = best
        fixed[i] = True
    for i in range(6):
        if not _is_nice(float(w[i])):
            w[i] = float(f"{w[i]:.4g}")
    info["snapped"] = [float(x) for x in w]
    src = _emit(w, less, spec)
    try:
        info["nodes"] = program_size(parse_program(src, spec.arg_names))["nodes"]
    except ProgramError:
        return None, info
    info["src"] = src
    info["passed"] = bool(check(src))
    return src, info
