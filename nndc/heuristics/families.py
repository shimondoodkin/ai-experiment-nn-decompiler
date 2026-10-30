"""F1 (ast5): formula families. Fit a·f(u) + d to the net, snap the parameters, add hinge terms in u.

Design: docs/design/2026-10-30-irrational-functions-design.md. The family list is the same for every net; the route
never learns which task a net is. Approximate: the checker decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from scipy.optimize import least_squares

from ..check import check_program
from ..lang import ProgramError, parse_program, program_size
from ..model import MLP
from ..spec import Spec
from .cleanup import join_terms, program_from_expr
from .snap import NICE_DENOMS, nice

SNAP_LEVELS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
MAX_TERMS = 4
KNOTS = 97
DROP_FRAC = 0.25
N_GRID = 2001
N_FIT_2D = 20_000
N_EXTREME = 200
N_SNAP_2D = 4000      # snapping refits in 2-D use a subsample (plus the extremes); errors are on the full sample
BIG = 1e3             # residual for a non-finite model output, in output ranges

FAMILIES_1D = ("identity", "square", "exp", "log", "sqrt", "reciprocal")
FAMILIES_2D = ("quadratic", "sqrt", "exp", "log", "reciprocal")
INNER = {1: ("b", "c"), 2: ("p0", "p1", "p2", "p3", "p4", "p5")}
QUAD_ATOMS = ("x0 * x0", "x1 * x1", "x0 * x1", "x0", "x1", None)


@dataclass
class Model:
    family: str
    dim: int
    names: list[str]
    values: np.ndarray
    fixed: np.ndarray                                   # fixed by the family, or snapped
    signs: list[int] = field(default_factory=list)      # hinge directions, one per k/t pair

    def copy(self) -> "Model":
        return Model(self.family, self.dim, list(self.names), self.values.copy(), self.fixed.copy(), list(self.signs))

    def get(self, v: np.ndarray, name: str) -> float:
        return v[self.names.index(name)]

    def inner(self, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            return v[0] * X[:, 0] + v[1]              # names start with b, c
        return _quad_matrix(X) @ v[:6]               # names start with p0..p5

    def hinge_var(self, v: np.ndarray, X: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        return X[:, 0] if self.dim == 1 else (self.inner(v, X) if u is None else u)

    def eval(self, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            u = self.inner(v, X)
            y = self.get(v, "a") * _f(self.family, u) + self.get(v, "d")
            h = self.hinge_var(v, X, u)
            for j, s in enumerate(self.signs):
                y = y + self.get(v, f"k{j}") * np.maximum(0.0, s * (h - self.get(v, f"t{j}")))
        return y


_QUAD_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _quad_matrix(X: np.ndarray) -> np.ndarray:
    """[x0², x1², x0·x1, x0, x1, 1] for X, cached by array identity (the route reuses a few sample arrays)."""
    hit = _QUAD_CACHE.get(id(X))
    if hit is not None and hit[0] is X:
        return hit[1]
    if len(_QUAD_CACHE) > 8:
        _QUAD_CACHE.clear()
    Q = np.stack(_quad_cols(X), axis=1)
    _QUAD_CACHE[id(X)] = (X, Q)
    return Q


def _f(family: str, u: np.ndarray) -> np.ndarray:
    if family in ("identity", "quadratic"):
        return u
    if family == "square":
        return u * u
    if family == "reciprocal":
        return 1.0 / u
    return getattr(np, family)(u)


def _new(family: str, dim: int, start: dict[str, float], fixed: set[str]) -> Model:
    names = [*INNER[dim], "a", "d"]
    return Model(family, dim, names, np.array([start.get(n, 0.0) for n in names], dtype=np.float64),
                 np.array([n in fixed for n in names]))


def _err(m: Model, v: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    r = np.abs(m.eval(v, X) - y)
    return float(np.max(np.where(np.isfinite(r), r, np.inf)))


def _refit(m: Model, X: np.ndarray, y: np.ndarray, R: float, nfev: int) -> np.ndarray:
    free = np.nonzero(~m.fixed)[0]
    v = m.values.copy()
    if not len(free) or len(X) <= len(free):
        return v

    def res(z: np.ndarray) -> np.ndarray:
        w = v.copy()
        w[free] = z
        return np.nan_to_num(m.eval(w, X) - y, nan=BIG * R, posinf=BIG * R, neginf=-BIG * R)
    try:
        z = least_squares(res, v[free], method="lm", max_nfev=nfev).x
    except (ValueError, np.linalg.LinAlgError):
        return v
    v[free] = z
    return v


def _lin(cols: list[np.ndarray], t: np.ndarray) -> np.ndarray | None:
    A = np.stack(cols, axis=1)
    if not np.isfinite(A).all() or not np.isfinite(t).all():
        return None
    return np.linalg.lstsq(A, t, rcond=None)[0]


def _quad_cols(X: np.ndarray) -> list[np.ndarray]:
    x0, x1 = X[:, 0], X[:, 1]
    return [x0 * x0, x1 * x1, x0 * x1, x0, x1, np.ones_like(x0)]


def _starts(family: str, dim: int, X: np.ndarray, y: np.ndarray, R: float) -> Iterator[tuple[dict, set]]:
    """Starting points (values, fixed names). Each family fixes the parameters it does not need (scale redundancy)."""
    with np.errstate(all="ignore"):
        if dim == 1:
            x = X[:, 0]
            lo, hi = float(x.min()), float(x.max())
            w = hi - lo
            one = np.ones_like(x)
            if family == "identity":
                c = _lin([x, one], y)
                if c is not None:
                    yield {"b": c[0], "c": c[1], "a": 1.0, "d": 0.0}, {"a", "d"}
                return
            if family == "square":
                A, B, C = np.polyfit(x, y, 2)
                A = A if abs(A) > 1e-9 else 1e-3
                yield {"b": 1.0, "c": B / (2 * A), "a": A, "d": C - B * B / (4 * A)}, {"b"}
                return
            if family == "exp":
                for b in (0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0):
                    b = b * 4 / w
                    c = _lin([np.exp(b * x), one], y)
                    if c is not None:
                        yield {"b": b, "c": 0.0, "a": c[0], "d": c[1]}, {"c"}
                return
            # log, sqrt, reciprocal: u = s·x + c, positive (or for reciprocal: of one sign) on the box
            for s in (1.0, -1.0):
                edge = -lo if s > 0 else hi
                for dl in ((0.0, 0.01, 0.1, 0.5, 2.0) if family == "sqrt" else (0.01, 0.1, 0.5, 2.0)):
                    for c0 in ((edge + dl * w,) if family != "reciprocal" else (edge + dl * w, edge - w - dl * w)):
                        u = s * x + c0
                        if family == "reciprocal":
                            if s < 0:
                                continue          # a/(-x + c) = -a/(x - c): the sign lives in a
                            c = _lin([1.0 / u, one], y)
                        else:
                            c = _lin([_f(family, u), one], y)
                        if c is not None:
                            yield {"b": s, "c": c0, "a": c[0], "d": c[1]}, {"b"}
            return
        cols = _quad_cols(X)
        if family == "quadratic":
            p = _lin(cols, y)
            if p is not None:
                yield {**{f"p{i}": p[i] for i in range(6)}, "a": 1.0, "d": 0.0}, {"a", "d"}
            return
        ymin, ymax = float(y.min()), float(y.max())
        for s in (1.0, -1.0):
            for dl in (0.0, 0.01, 0.1, 0.5):
                d = ymin - dl * R if s > 0 else ymax + dl * R
                if family == "exp":
                    if dl == 0.0:
                        continue
                    p = _lin(cols, np.log((y - d) / s))
                    if p is not None:
                        yield {**{f"p{i}": p[i] for i in range(5)}, "p5": 0.0, "a": s * np.exp(p[5]), "d": d}, {"p5"}
                elif family == "sqrt":
                    p = _lin(cols, ((y - d) / s) ** 2)
                    if p is not None:
                        yield {**{f"p{i}": p[i] for i in range(6)}, "a": s, "d": d}, {"a"}
                elif family == "reciprocal":
                    if dl == 0.0:
                        continue
                    p = _lin(cols, s / (y - d))
                    if p is not None:
                        yield {**{f"p{i}": p[i] for i in range(6)}, "a": s, "d": d}, {"a"}
        if family == "log":
            ym = float(np.median(y))
            for a in (0.5, 1.0, 2.0, -0.5, -1.0, -2.0):
                a = a * R
                p = _lin(cols, np.exp((y - ym) / a))
                if p is not None:
                    yield {**{f"p{i}": p[i] for i in range(6)}, "a": a, "d": ym}, {"d"}


def _fit_family(family: str, dim: int, X: np.ndarray, y: np.ndarray, Xs: np.ndarray, ys: np.ndarray,
                R: float) -> tuple[Model | None, float]:
    """A short fit from every start on the subsample (Xs, ys), then the best one in full on (X, y)."""
    best, best_err = None, np.inf
    for start, fixed in _starts(family, dim, Xs, ys, R):
        m = _new(family, dim, start, fixed)
        m.values = _refit(m, Xs, ys, R, 300)
        e = _err(m, m.values, Xs, ys)
        if e < best_err:
            best, best_err = m, e
    if best is not None:
        best.values = _refit(best, X, y, R, 4000)
        best_err = _err(best, best.values, X, y)
    return best, best_err


def _is_nice(c: float) -> bool:
    return any(round(c * d) / d == c for d in NICE_DENOMS)


def _snap(m: Model, X: np.ndarray, y: np.ndarray, Xs: np.ndarray, ys: np.ndarray, R: float, tol: float) -> Model:
    """Sequential snap with refit (spike): each round, the free parameter that snaps at the coarsest level, then a
    refit of the other free ones; a level is accepted only if the max error stays within tol. Parameters that never
    snap are rounded to 4 significant digits."""
    m = m.copy()
    hopeless: set[int] = set()       # failed at every level in an earlier round: not tried again
    while (~m.fixed).any():
        best = None
        for i in np.nonzero(~m.fixed)[0]:
            if i in hopeless:
                continue
            tried: set[float] = set()
            for lev in SNAP_LEVELS:
                if best is not None and lev <= best[0]:
                    break
                c = nice(float(m.values[i]), lev * max(1.0, abs(float(m.values[i]))))
                if not _is_nice(c) or c in tried:   # nice() returns the value itself when nothing nice is in reach
                    continue
                tried.add(c)
                t = m.copy()
                t.values[i] = c
                t.fixed[i] = True
                if _err(t, t.values, X, y) > tol:
                    t.values = _refit(t, Xs, ys, R, 100)
                    if _err(t, t.values, X, y) > tol:
                        continue
                best = (lev, t)
                break
            else:
                if best is None or best[1] is None:
                    hopeless.add(i)
        if best is None:
            break
        m = best[1]
    for i, x in enumerate(m.values):     # unsnapped and structural values: 4 significant digits
        if not _is_nice(float(x)):
            m.values[i] = float(f"{x:.4g}")
    return m


def _wrap(s: str) -> str:
    return s if s.replace("_", "").isalnum() else f"({s})"


def _emit(m: Model, spec: Spec) -> str:
    v = m.values
    if m.dim == 1:
        u = join_terms([(m.get(v, "b"), "x0"), (m.get(v, "c"), None)])
        h = "x0"
    else:
        u = join_terms([(m.get(v, f"p{i}"), atom) for i, atom in enumerate(QUAD_ATOMS)])
        h = "u"
    derived = {"u": u} if m.dim == 2 and m.signs else None
    uu = "u" if derived else u
    a, d = m.get(v, "a"), m.get(v, "d")
    if m.family in ("identity", "quadratic"):
        if m.dim == 1:
            items = [(m.get(v, "b"), "x0")]
            d = d + m.get(v, "c")
        else:
            items = [(a, _wrap(uu))] if derived else \
                [(m.get(v, f"p{i}"), atom) for i, atom in enumerate(QUAD_ATOMS) if atom]
            if not derived:
                d = d + m.get(v, "p5")
    elif m.family == "square":
        items = [(a, f"{_wrap(uu)} * {_wrap(uu)}")]
    elif m.family == "reciprocal":
        items = [(1.0 if a > 0 else -1.0, f"{join_terms([(abs(a), None)])} / {_wrap(uu)}")] if a else []
    else:
        items = [(a, f"{m.family}({uu})")]
    for j, s in enumerate(m.signs):
        t = m.get(v, f"t{j}")
        arg = join_terms([(1.0, h), (-t, None)]) if s > 0 else join_terms([(t, None), (-1.0, h)])
        items.append((m.get(v, f"k{j}"), f"relu({arg})"))
    return program_from_expr(join_terms(items + [(d, None)]), spec, derived)


def _add_hinge(m: Model, X: np.ndarray, y: np.ndarray, R: float) -> Model | None:
    """The knot (and direction) whose hinge column, refit linearly with the free a, d and k's, leaves the smallest
    max error; then a nonlinear refit of every free parameter."""
    v = m.values
    with np.errstate(all="ignore"):
        fu = _f(m.family, m.inner(v, X))
        h = m.hinge_var(v, X)
    lin = []                                            # (parameter index, column) of the free linear parameters
    if not m.fixed[m.names.index("a")]:
        lin.append((m.names.index("a"), fu))
    if not m.fixed[m.names.index("d")]:
        lin.append((m.names.index("d"), np.ones(len(X))))
    for j, s in enumerate(m.signs):
        if not m.fixed[m.names.index(f"k{j}")]:
            lin.append((m.names.index(f"k{j}"), np.maximum(0.0, s * (h - m.get(v, f"t{j}")))))
    base = v.copy()
    for i, _ in lin:
        base[i] = 0.0
    rest = y - m.eval(base, X)
    if not np.isfinite(rest).all():
        return None
    hlo, hhi = float(h.min()), float(h.max())
    best = None
    for t in np.linspace(hlo, hhi, KNOTS)[1:-1]:
        for s in (1, -1):
            col = np.maximum(0.0, s * (h - t))
            c = _lin([c for _, c in lin] + [col], rest)
            if c is None:
                continue
            e = float(np.max(np.abs(np.stack([c for _, c in lin] + [col], axis=1) @ c - rest)))
            if best is None or e < best[0]:
                best = (e, t, s, c)
    if best is None:
        return None
    _, t, s, c = best
    j = len(m.signs)
    n = m.copy()
    n.names += [f"k{j}", f"t{j}"]
    n.values = np.append(n.values, [c[-1], t])
    n.fixed = np.append(n.fixed, [False, False])
    n.signs.append(s)
    for (i, _), ci in zip(lin, c[:-1]):
        n.values[i] = ci
    n.values = _refit(n, X, y, R, 500)
    return n


def _fit_sample(spec: Spec, predict: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(spec.inputs) == 1:
        i = spec.inputs[0]
        X = np.linspace(i.lo, i.hi, N_GRID)[:, None]
        return X, np.asarray(predict(X), dtype=np.float64)
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(i.lo, i.hi, N_FIT_2D) for i in spec.inputs], axis=1)
    y = np.asarray(predict(X), dtype=np.float64)
    ext = np.argsort(-np.abs(y - np.median(y)), kind="stable")[:N_EXTREME]
    return np.vstack([X, X[ext]]), np.concatenate([y, y[ext]])


def family_candidates(spec: Spec, mlp: MLP | None, predict: Callable[[np.ndarray], np.ndarray] | None = None,
                      check: Callable[[str], bool] | None = None) -> tuple[list[tuple[str, str]], list[dict]]:
    """Passing family programs as (family, src), and one info row per family tried. predict and check are test
    hooks (default: the net's output, and the checker's snap verdict against the net)."""
    if spec.output != "regression" or len(spec.inputs) not in (1, 2):
        return [], []
    if predict is None:
        predict = lambda X: mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)   # noqa: E731
    if check is None:
        check = lambda src: check_program(src, mlp, spec).snap_pass   # noqa: E731
    X, y = _fit_sample(spec, predict)
    R = float(y.max() - y.min())
    if not np.isfinite(R) or R <= 1e-12:
        return [], []
    tol = spec.snap_tol_frac * R
    dim = len(spec.inputs)
    if dim == 2:
        k = len(X) - N_EXTREME
        Xs, ys = np.vstack([X[:N_SNAP_2D], X[k:]]), np.concatenate([y[:N_SNAP_2D], y[k:]])
    else:
        Xs, ys = X, y
    passing, info = [], []
    for family in (FAMILIES_1D if dim == 1 else FAMILIES_2D):
        m, e = _fit_family(family, dim, X, y, Xs, ys, R)
        row = {"family": family, "fit_err": e / R if m is not None else float("inf"), "snapped_err": None,
               "terms": 0, "passed": False, "nodes": None, "src": None, "params": None}
        info.append(row)
        if m is None or not e / R <= DROP_FRAC:
            continue
        for terms in range(MAX_TERMS + 1):
            if terms:
                n = _add_hinge(m, X, y, R)
                if n is None:
                    break
                m = n
            m = _snap(m, X, y, Xs, ys, R, tol)
            src = _emit(m, spec)
            try:
                nodes = program_size(parse_program(src, spec.arg_names))["nodes"]
            except ProgramError:
                break
            ok = bool(check(src))
            row.update(snapped_err=_err(m, m.values, X, y) / R, terms=terms, passed=ok, nodes=nodes, src=src,
                       params={n: float(x) for n, x in zip(m.names, m.values)})
            if ok:
                passing.append((family, src))
                break
    return passing, info
