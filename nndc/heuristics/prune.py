"""F3 (ast5): prune with a minimax output refit, for one-hidden-layer regression nets when no formula passes.

Design: docs/design/2026-10-30-irrational-functions-design.md, amendment "route changes distilled from the agent
pass", change 2 (change 3, the joint fold, is `_foldable`). HEURISTICS (runs_irr/HEURISTICS.md): minimax LP refit of
the output layer (sigmoid bump/seed1, sigmoid dist/seed1, relu bump/seed0, tanh bump/seed2); greedy backward
elimination with a free affine term (sigmoid dist/seed0, tanh dist/seed0, relu_wide bump/seed0); one parameter
snapped at a time with a refit, hidden first, outputs last and finer (tanh dist/seed1, sigmoid bump/seed2).

Blind: the net, the blind spec and the checker only. tanh nets arrive rewritten as sigmoid nets (`refactor`).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..model import MLP
from ..spec import Spec
from .cleanup import join_terms, program_from_expr
from .families import SNAP_LEVELS, _is_nice, checker_hook
from .sigmoid import minimax
from .snap import nice

PRUNE_BUDGET = 0.8        # every step keeps the max error on the fit points within this share of the tolerance
N_GRID_1D = 2001
N_GRID_2D = 61            # 61 × 61 grid; the box border is on it
N_SCREEN = 3              # removals screened by least squares; the best N_SCREEN are confirmed by LP
OUT_SNAP_LEVELS = (0.01, 0.005, 0.001)
PRUNE_RETRIES = 2         # rounds of adding the checker's counterexamples after a failure
DIGITS = 5                # significant digits of the weights that do not snap
FOLD_WIDTH = 1.0          # change 3: sigmoid units whose pre-activation spans at most this over the box are folded
                          # (HEURISTICS: sigmoid dist/seed1, tanh dist/seed0, tanh dist/seed2, tanh exp/seed0)


def _grid(spec: Spec) -> np.ndarray:
    if len(spec.inputs) == 1:
        i = spec.inputs[0]
        return np.linspace(i.lo, i.hi, N_GRID_1D)[:, None]
    a, b = spec.inputs
    g0, g1 = np.meshgrid(np.linspace(a.lo, a.hi, N_GRID_2D), np.linspace(b.lo, b.hi, N_GRID_2D))
    return np.stack([g0.ravel(), g1.ravel()], axis=1)


def _act(act: str, z: np.ndarray) -> np.ndarray:
    if act == "relu":
        return np.maximum(z, 0.0)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def _extras(X: np.ndarray, names: list[str], quad: bool) -> list[tuple[str | None, np.ndarray]]:
    """(atom, column) of the free terms: the inputs, with quad the quadratic atoms, and the constant last."""
    out: list[tuple[str | None, np.ndarray]] = [(n, X[:, i]) for i, n in enumerate(names)]
    if quad:
        out.append((f"{names[0]} * {names[0]}", X[:, 0] ** 2))
        if len(names) == 2:
            out += [(f"{names[1]} * {names[1]}", X[:, 1] ** 2), (f"{names[0]} * {names[1]}", X[:, 0] * X[:, 1])]
    return out + [(None, np.ones(len(X)))]


def _foldable(mlp: MLP, spec: Spec) -> list[int]:
    """Units the joint fold replaces by free terms: sigmoid units whose pre-activation range over the box is at most
    FOLD_WIDTH (near-linear or saturated); ReLU units dead or always on over the box (exactly 0 or affine)."""
    W, b = mlp.weights[0].astype(np.float64), mlp.biases[0].astype(np.float64)
    lo = np.array([i.lo for i in spec.inputs])
    hi = np.array([i.hi for i in spec.inputs])
    zlo = b + np.minimum(W * lo, W * hi).sum(axis=1)
    zhi = b + np.maximum(W * lo, W * hi).sum(axis=1)
    if mlp.acts[0] == "relu":
        return [int(u) for u in np.nonzero((zlo >= 0) | (zhi <= 0))[0]]
    return [int(u) for u in np.nonzero(zhi - zlo <= FOLD_WIDTH)[0]]


def _lp(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    c, e = minimax(A, y)
    if not np.isfinite(e):
        c, e = minimax(A, y, method="highs-ipm")
    return c, e


def _lp_fixed(A: np.ndarray, y: np.ndarray, fixed: dict[int, float]) -> tuple[np.ndarray, float]:
    """The minimax output fit with the columns in fixed held at their values."""
    c = np.zeros(A.shape[1])
    rhs = y.copy()
    for i, v in fixed.items():
        c[i] = v
        rhs = rhs - v * A[:, i]
    free = [i for i in range(A.shape[1]) if i not in fixed]
    if not free:
        return c, float(np.abs(rhs).max())
    cf, e = _lp(A[:, free], rhs)
    c[free] = cf
    return c, e


def _round(v: float) -> float:
    return v if _is_nice(v) else float(f"{v:.{DIGITS}g}")


def _snap_outputs(A: np.ndarray, y: np.ndarray, c: np.ndarray, e: float, budget: float) -> tuple[np.ndarray, float]:
    """Each output weight in turn to the coarsest nice value at OUT_SNAP_LEVELS whose re-solve of the others stays
    within the budget; the rest rounded to DIGITS."""
    fixed: dict[int, float] = {}
    for i in range(A.shape[1]):
        for lev in OUT_SNAP_LEVELS:
            v = nice(float(c[i]), lev * max(1.0, abs(float(c[i]))))
            if not _is_nice(v):
                continue
            c2, e2 = _lp_fixed(A, y, fixed | {i: v})
            if e2 <= budget:
                fixed[i], c, e = v, c2, e2
                break
    return np.array([_round(float(x)) for x in c]), e


def _emit(spec: Spec, act: str, W: np.ndarray, b: np.ndarray, units: list[int], c: np.ndarray,
          atoms: list[str | None]) -> str:
    names = spec.arg_names
    derived, items = {}, []
    for pos, u in enumerate(units):
        if c[pos] == 0:
            continue
        z = join_terms([(float(W[u, i]), n) for i, n in enumerate(names)] + [(float(b[u]), None)])
        derived[f"h{u}"] = f"{act}({z})"
        items.append((float(c[pos]), f"h{u}"))
    items += [(float(ci), atom) for ci, atom in zip(c[len(units):], atoms)]
    return program_from_expr(join_terms(items), spec, derived or None)


def prune_candidate(spec: Spec, mlp: MLP, check: Callable[[str], tuple[bool, np.ndarray]] | None = None
                    ) -> tuple[str | None, dict]:
    """The pruned, refit and snapped net as a program, or None; and an info dict (kept and folded units, max error
    on the fit points as a share of the range, retries, a note when not applicable). check is a test hook (default:
    the checker against mlp, with its counterexamples). A numerical failure (lstsq, linprog) gives None and a note."""
    info: dict = {"kept": [], "folded": [], "err": None, "retries": 0, "passed": False, "note": None}
    try:
        return _prune(spec, mlp, check, info)
    except (np.linalg.LinAlgError, ValueError) as e:
        info["note"] = f"numerical failure: {type(e).__name__}: {e}"
        return None, info


def _prune(spec: Spec, mlp: MLP, check: Callable[[str], tuple[bool, np.ndarray]] | None, info: dict
           ) -> tuple[str | None, dict]:
    if (spec.output != "regression" or len(spec.inputs) not in (1, 2) or len(mlp.weights) != 2
            or mlp.weights[1].shape[0] != 1):
        info["note"] = "needs a one-hidden-layer regression net with 1 or 2 inputs"
        return None, info
    act = mlp.acts[0]
    if act not in ("relu", "sigmoid"):
        info["note"] = f"activation {act}"
        return None, info
    if check is None:
        check = checker_hook(mlp, spec)
    names = spec.arg_names
    W = mlp.weights[0].astype(np.float64).copy()
    b = mlp.biases[0].astype(np.float64).copy()
    X = _grid(spec)
    y = mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)
    R = float(np.ptp(y))
    if not np.isfinite(R) or R <= 1e-12:
        info["note"] = "constant net"
        return None, info
    budget = PRUNE_BUDGET * spec.snap_tol_frac * R
    folded = _foldable(mlp, spec)                  # change 3: their sum becomes free terms, re-solved jointly
    units = [u for u in range(W.shape[0]) if u not in folded]
    quad = act == "sigmoid" and bool(folded)       # a folded sigmoid unit may curve: allow the quadratic atoms
    info["folded"] = folded
    keep_atoms = [a for a, _ in _extras(X, names, quad)]   # free terms still in the fit; the constant is last

    def design(X: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
        ex = [(a, col) for a, col in _extras(X, names, quad) if a in keep_atoms]
        H = _act(act, X @ W[units].T + b[units]) if units else np.zeros((len(X), 0))
        return np.hstack([H, np.stack([col for _, col in ex], axis=1)]), [a for a, _ in ex]

    A, atoms = design(X)
    c, e = _lp(A, y)
    if not np.isfinite(e):
        info["note"] = "LP failed (highs and highs-ipm)"
        return None, info

    while A.shape[1] > 1:                          # greedy backward elimination of units and free terms
        screen = []
        for j in range(A.shape[1] - 1):            # the constant column (last) always stays
            Aj = np.delete(A, j, axis=1)
            cj = np.linalg.lstsq(Aj, y, rcond=None)[0]
            screen.append(float(np.abs(Aj @ cj - y).max()))
        best = None
        for j in np.argsort(screen, kind="stable")[:N_SCREEN]:
            cj, ej = _lp(np.delete(A, j, axis=1), y)
            if best is None or ej < best[0]:
                best = (ej, int(j), cj)
        if best is None or not best[0] <= budget:
            break
        e, j, c = best
        if j < len(units):
            units.pop(j)
        else:
            keep_atoms.pop(j - len(units))
        A = np.delete(A, j, axis=1)

    for pos, u in enumerate(units):                # hidden weights, one at a time, coarsest level first
        for p in range(len(names) + 1):
            v = float(W[u, p] if p < len(names) else b[u])
            for lev in SNAP_LEVELS:
                s = nice(v, lev * max(1.0, abs(v)))
                if not _is_nice(s) or s == v:
                    continue
                w2, b2 = W[u].copy(), float(b[u])
                if p < len(names):
                    w2[p] = s
                else:
                    b2 = s
                A2 = A.copy()
                A2[:, pos] = _act(act, X @ w2 + b2)
                c2, e2 = _lp(A2, y)
                if e2 <= budget:
                    W[u], b[u], A, c, e = w2, b2, A2, c2, e2
                    break
    W = np.vectorize(_round)(W)
    b = np.vectorize(_round)(b)

    src = None
    while True:
        A, atoms = design(X)
        c, e = _lp(A, y)
        if not np.isfinite(e):
            info["note"] = "LP failed (highs and highs-ipm)"
            return None, info
        c, e = _snap_outputs(A, y, c, e, budget)
        src = _emit(spec, act, W, b, units, c, atoms)
        ok, cex = check(src)
        if ok or info["retries"] >= PRUNE_RETRIES or not len(cex):
            break
        cex = np.asarray(cex, dtype=np.float64).reshape(-1, len(names))
        X = np.vstack([X, cex])
        y = np.concatenate([y, mlp.forward(cex.astype(np.float32))[:, 0].astype(np.float64)])
        info["retries"] += 1
    info.update(kept=list(units), atoms=list(keep_atoms), err=e / R, passed=bool(ok))
    return src, info
