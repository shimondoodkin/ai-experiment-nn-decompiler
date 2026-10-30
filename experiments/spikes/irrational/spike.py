"""Throwaway spike: irrational functions (exp, log, sqrt, x^2, 1/x, circle, distance, bump, product).

Per task x act (relu, sigmoid), seed 0:
  1. train a net with the repo's train();
  2. how the net models it: kink / unit-centre positions (1-D), boundary polygon (circle), unit lines (2-D);
  3. formula + terms: fit families a*f(b*x+c)+d to the NET, snap constants, check within 2% of the net's range;
     if it fails, add hinge correction terms greedily until it passes;
  4. what ast4 makes of it today (nodes, route).
Writes out/<task>_<act>.json. Run: uv run python spike.py [task ...]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from experiments.tasks import Task, _inputs
from experiments.train import train
from nndc.ast_tool import refactor
from nndc.check import build_input_set, check_program, net_output
from nndc.dump import dump_program
from nndc.heuristics.snap import nice
from nndc.lang import parse_program, program_size
from nndc.spec import Spec

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

R2 = lambda X: X[:, 0] ** 2 + X[:, 1] ** 2
TASKS = {
    "exp": (Task("exp", Spec("exp", "regression", _inputs(("x", "float", -2, 2))),
                 lambda X: np.exp(X[:, 0]), "", hidden=[8], handcrafted=None, steps=6000), "exp(x0)"),
    "log": (Task("log", Spec("log", "regression", _inputs(("x", "float", 0.1, 5))),
                 lambda X: np.log(X[:, 0]), "", hidden=[8], handcrafted=None, steps=6000), "log(x0)"),
    "sqrt": (Task("sqrt", Spec("sqrt", "regression", _inputs(("x", "float", 0, 4))),
                  lambda X: np.sqrt(X[:, 0]), "", hidden=[8], handcrafted=None, steps=6000), "sqrt(x0)"),
    "square": (Task("square", Spec("square", "regression", _inputs(("x", "float", -2, 2))),
                    lambda X: X[:, 0] ** 2, "", hidden=[8], handcrafted=None, steps=6000), "x0*x0"),
    "recip": (Task("recip", Spec("recip", "regression", _inputs(("x", "float", 0.5, 4))),
                   lambda X: 1 / X[:, 0], "", hidden=[8], handcrafted=None, steps=6000), "1/x0"),
    "circle": (Task("circle", Spec("circle", "binary", _inputs(("a", "float", -3, 3), ("b", "float", -3, 3))),
                    lambda X: (R2(X) < 4).astype(int), "", hidden=[16], handcrafted=None, steps=6000),
               "x0*x0 + x1*x1 < 4"),
    "dist": (Task("dist", Spec("dist", "regression", _inputs(("a", "float", -3, 3), ("b", "float", -3, 3))),
                  lambda X: np.sqrt(R2(X)), "", hidden=[16], handcrafted=None, steps=6000), "sqrt(x0*x0 + x1*x1)"),
    "bump": (Task("bump", Spec("bump", "regression", _inputs(("a", "float", -3, 3), ("b", "float", -3, 3))),
                  lambda X: np.exp(-R2(X)), "", hidden=[16], handcrafted=None, steps=6000), "exp(-(x0*x0 + x1*x1))"),
    "product": (Task("product", Spec("product", "regression", _inputs(("a", "float", -2, 2), ("b", "float", -2, 2))),
                     lambda X: X[:, 0] * X[:, 1], "", hidden=[16], handcrafted=None, steps=6000), "x0*x1"),
}

# families: name -> (fn(params, X) , n params, init list, printer)
def _safe(f):
    def g(p, X):
        with np.errstate(all="ignore"):
            return f(p, X)
    return g

FAM1 = {
    "exp":    (_safe(lambda p, X: p[0] * np.exp(p[1] * X[:, 0]) + p[2]), [[1, 1, 0], [1, 0.5, 0], [-1, -1, 0]],
               "{0}*exp({1}*x0) + {2}"),
    "log":    (_safe(lambda p, X: p[0] * np.log(X[:, 0] + p[1]) + p[2]), [[1, 0, 0], [1, 1, 0], [-1, 0.5, 0]],
               "{0}*log(x0 + {1}) + {2}"),
    "sqrt":   (_safe(lambda p, X: p[0] * np.sqrt(X[:, 0] + p[1]) + p[2]), [[1, 0, 0], [1, 1, 0], [-1, 2, 0]],
               "{0}*sqrt(x0 + {1}) + {2}"),
    "quad":   (_safe(lambda p, X: p[0] * X[:, 0] ** 2 + p[1] * X[:, 0] + p[2]), [[1, 0, 0]],
               "{0}*x0*x0 + {1}*x0 + {2}"),
    "recip":  (_safe(lambda p, X: p[0] / (X[:, 0] + p[1]) + p[2]), [[1, 0, 0], [1, 1, 0], [-1, 0.5, 0]],
               "{0}/(x0 + {1}) + {2}"),
}
FAM2 = {
    "quad_r2":  (_safe(lambda p, X: p[0] * R2(X) + p[1]), [[1, 0]], "{0}*(x0*x0 + x1*x1) + {1}"),
    "sqrt_r2":  (_safe(lambda p, X: p[0] * np.sqrt(R2(X) + p[1]) + p[2]), [[1, 0, 0], [1, 1, 0]],
                 "{0}*sqrt(x0*x0 + x1*x1 + {1}) + {2}"),
    "exp_r2":   (_safe(lambda p, X: p[0] * np.exp(p[1] * R2(X)) + p[2]), [[1, -1, 0], [1, -0.5, 0]],
                 "{0}*exp({1}*(x0*x0 + x1*x1)) + {2}"),
    "product":  (_safe(lambda p, X: p[0] * X[:, 0] * X[:, 1] + p[1]), [[1, 0]], "{0}*x0*x1 + {1}"),
    "quad_gen": (_safe(lambda p, X: p[0] * X[:, 0] ** 2 + p[1] * X[:, 1] ** 2 + p[2] * X[:, 0] * X[:, 1]
                       + p[3] * X[:, 0] + p[4] * X[:, 1] + p[5]), [[0, 0, 0, 0, 0, 0]],
                 "{0}*x0*x0 + {1}*x1*x1 + {2}*x0*x1 + {3}*x0 + {4}*x1 + {5}"),
}
SNAP_LEVELS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.001)


def fit(fn, inits, X, y):
    best = None
    for p0 in inits:
        try:
            r = least_squares(lambda p: np.nan_to_num(fn(p, X) - y, nan=1e3, posinf=1e3, neginf=-1e3), p0,
                              method="lm", max_nfev=4000)
        except Exception:
            continue
        e = float(np.max(np.abs(np.nan_to_num(fn(r.x, X) - y, nan=1e9))))
        if best is None or e < best[1]:
            best = (r.x, e)
    return best


def maxerr(fn, p, X, y):
    v = fn(np.asarray(p, float), X)
    return float(np.max(np.abs(np.where(np.isfinite(v), v - y, 1e9))))


def snap_params(fn, p, X, y, tol):
    """Sequential snap with refit: params in order; each takes the coarsest nice value that keeps max err <= tol."""
    p = list(map(float, p))
    fixed = [False] * len(p)
    for _ in range(len(p)):
        # snap the param that is closest to a nice value (relative), first
        best = None
        for i in range(len(p)):
            if fixed[i]:
                continue
            for lev in SNAP_LEVELS:
                c = nice(p[i], lev * max(1.0, abs(p[i])))
                q = p.copy(); q[i] = c
                free = [j for j in range(len(p)) if not fixed[j] and j != i]
                if free:
                    def res(z):
                        qq = q.copy()
                        for j, v in zip(free, z):
                            qq[j] = v
                        return np.nan_to_num(fn(np.asarray(qq), X) - y, nan=1e3, posinf=1e3, neginf=-1e3)
                    z = least_squares(res, [q[j] for j in free], method="lm", max_nfev=2000).x
                    for j, v in zip(free, z):
                        q[j] = float(v)
                e = maxerr(fn, q, X, y)
                if e <= tol:
                    if best is None or lev > best[0]:
                        best = (lev, i, q)
                    break
        if best is None:
            break
        _, i, q = best
        p = q; fixed[i] = True
    return p, fixed, maxerr(fn, p, X, y)


def fmtp(v):
    return f"{v:g}" if v == round(v, 6) else f"{v:.4g}"


def hinge_terms(X, r, tol, max_terms=12, ncand=97):
    """Greedy: r ~ c0 + c1*x0 + sum c_k*max(0, s*(x0 - t_k)); returns n terms needed (1-D only)."""
    x = X[:, 0]
    lo, hi = x.min(), x.max()
    knots = np.linspace(lo, hi, ncand)[1:-1]
    cols = [np.ones_like(x), x]
    chosen = []
    for k in range(max_terms + 1):
        A = np.stack(cols, 1)
        c, *_ = np.linalg.lstsq(A, r, rcond=None)
        e = float(np.max(np.abs(A @ c - r)))
        if e <= tol:
            return k, chosen, e
        if k == max_terms:
            return None, chosen, e
        best = None
        for t in knots:
            col = np.maximum(0, x - t)
            A2 = np.stack(cols + [col], 1)
            c2, *_ = np.linalg.lstsq(A2, r, rcond=None)
            e2 = float(np.max(np.abs(A2 @ c2 - r)))
            if best is None or e2 < best[0]:
                best = (e2, t, col)
        chosen.append(round(float(best[1]), 3)); cols.append(best[2])


def boundary_by_angle(mlp, spec, n=720, rmax=4.2):
    """For each angle, the radius where the net's binary decision flips (first crossing from 0 outward)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = []
    for t in th:
        d = np.array([math.cos(t), math.sin(t)])
        r_grid = np.linspace(0, rmax, 400)
        P = r_grid[:, None] * d
        inbox = (np.abs(P) <= 3).all(1)
        z = mlp.forward(P.astype(np.float32))[:, 0] - spec.threshold
        s = (z > 0) & inbox
        idx = np.nonzero(s[:-1] != s[1:])[0]
        if len(idx) == 0:
            rs.append(np.nan); continue
        a, b = r_grid[idx[0]], r_grid[idx[0] + 1]
        for _ in range(40):
            m = (a + b) / 2
            zm = mlp.forward((m * d)[None].astype(np.float32))[0, 0] - spec.threshold
            if ((zm > 0) == s[idx[0]]):
                a = m
            else:
                b = m
        rs.append((a + b) / 2)
    return th, np.array(rs)


def analyse(name, act):
    task, intended = TASKS[name]
    t0 = time.time()
    if act == "relu_wide":
        task = Task(**{**task.__dict__, "hidden": [4 * h for h in task.hidden], "steps": 3 * task.steps})
    mlp, log = train(task, 0, act=act.replace("_wide", ""))
    spec = task.spec
    blind = Spec.from_json(spec.blind())
    out = {"task": name, "act": act, "intended": intended, "learned": log["learned"], "quality": log["quality"],
           "train_s": round(time.time() - t0, 1)}
    X, label = build_input_set(mlp, spec, 0)
    n_dec, n_score = net_output(mlp, spec, X)
    W1, b1 = mlp.weights[0], mlp.biases[0]
    W2 = mlp.weights[1]
    d = len(spec.inputs)

    # ---- how the net models it
    if d == 1:
        lo, hi = spec.inputs[0].lo, spec.inputs[0].hi
        w, b = W1[:, 0].astype(float), b1.astype(float)
        cen = -b / np.where(w == 0, np.nan, w)
        units = []
        for i in range(len(b)):
            units.append({"centre": round(float(cen[i]), 3), "in_box": bool(lo < cen[i] < hi),
                          "w": round(float(w[i]), 3), "out": round(float(W2[0, i]), 3)})
        out["units"] = sorted(units, key=lambda u: u["centre"] if np.isfinite(u["centre"]) else 1e9)
        # piecewise-linear optimal knot spacing for comparison: density ~ sqrt|f''|
        xs = np.linspace(lo, hi, 4001)
        f = task.target(xs[:, None])
        f2 = np.abs(np.gradient(np.gradient(f, xs), xs))
        dens = np.sqrt(f2); cum = np.cumsum(dens); cum /= cum[-1]
        k = sum(u["in_box"] for u in units)
        out["n_kinks_in_box"] = k
        out["optimal_knots_same_count"] = [round(float(np.interp((j + 0.5) / k, cum, xs)), 3) for j in range(k)] if k else []
    elif name == "circle":
        th, rs = boundary_by_angle(mlp, spec)
        ok = np.isfinite(rs)
        out["boundary_found_angles"] = int(ok.sum())
        out["radius_min_max_mean"] = [round(float(np.nanmin(rs)), 3), round(float(np.nanmax(rs)), 3), round(float(np.nanmean(rs)), 3)]
        P = rs[ok, None] * np.stack([np.cos(th[ok]), np.sin(th[ok])], 1)
        pats = (mlp.preactivations(P.astype(np.float32))[0] > 0) if act.startswith("relu") else None
        if pats is not None:
            changes = int((pats[1:] != pats[:-1]).any(1).sum() + (pats[0] != pats[-1]).any())
            out["boundary_pieces_relu"] = changes
        # least-squares circle fit to the boundary points
        A = np.c_[2 * P, np.ones(len(P))]
        sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
        cx, cy, c = sol
        out["circle_fit"] = {"cx": round(float(cx), 4), "cy": round(float(cy), 4),
                             "r": round(float(math.sqrt(c + cx * cx + cy * cy)), 4)}
        # unit lines: distance from origin and angle
        lines = []
        for i in range(len(b1)):
            wv = W1[i]
            nrm = float(np.linalg.norm(wv))
            lines.append({"dist": round(float(abs(b1[i]) / nrm), 3) if nrm else None,
                          "angle_deg": round(float(np.degrees(np.arctan2(wv[1], wv[0]))) % 180, 1),
                          "steep": round(nrm, 2)})
        out["unit_lines"] = sorted(lines, key=lambda l: l["angle_deg"])
    else:
        lines = []
        for i in range(len(b1)):
            wv = W1[i]
            nrm = float(np.linalg.norm(wv))
            lines.append({"dist": round(float(abs(b1[i]) / nrm), 3) if nrm else None,
                          "angle_deg": round(float(np.degrees(np.arctan2(wv[1], wv[0]))) % 180, 1),
                          "steep": round(nrm, 2)})
        out["unit_lines"] = sorted(lines, key=lambda l: l["angle_deg"])

    # ---- formula + terms against the net
    if spec.output == "regression":
        y = n_dec
        rng_ = float(y.max() - y.min())
        tol = spec.snap_tol_frac * rng_
        out["net_range"] = round(rng_, 4); out["tol"] = round(tol, 5)
        true = task.target(X)
        out["intended_err_frac"] = round(float(np.max(np.abs(true - y))) / rng_, 4)
        fams = FAM1 if d == 1 else FAM2
        # fit on a small set (grid / sample + the input set's extremes); final errors on the full input set
        if d == 1:
            Xf = np.linspace(spec.inputs[0].lo, spec.inputs[0].hi, 2001)[:, None]
        else:
            sel = np.random.default_rng(0).choice(len(X), 20000, replace=False)
            Xf = np.vstack([X[sel], X[np.argsort(-np.abs(y - y.mean()))[:200]]])
        yf = net_output(mlp, spec, Xf)[0]
        res = {}
        for fam, (fn, inits, pr) in fams.items():
            r = fit(fn, inits, Xf, yf)
            if r is None:
                continue
            p, _ = r
            e = maxerr(fn, p, X, y)
            entry = {"fit_err_frac": round(e / rng_, 4), "fit": pr.format(*[fmtp(v) for v in p])}
            if e <= tol:
                ps, fixed, _ = snap_params(fn, p, Xf, yf, tol)
                es = maxerr(fn, ps, X, y)
                entry.update(snapped=pr.format(*[fmtp(v) for v in ps]), snapped_all=all(fixed),
                             snapped_err_frac=round(es / rng_, 4), snapped_pass=es <= tol)
            if d == 1 and e / rng_ < 0.25:
                resid = yf - fn(np.asarray(p), Xf)
                k, knots, ek = hinge_terms(Xf, resid, tol)
                entry["hinges_to_pass"] = k; entry["hinge_knots"] = knots
            res[fam] = entry
        # pure piecewise linear: hinges from scratch (1-D)
        if d == 1:
            k, knots, ek = hinge_terms(Xf, yf, tol, max_terms=16)
            out["linear_only_hinges_to_pass"] = k
        out["families"] = res
    else:
        r = check_program(f"def net(x0, x1):\n    return {intended}\n", mlp, blind)
        out["intended_check"] = {"snap_pass": r.snap_pass, "exact_pass": r.exact_pass,
                                 "outside": r.disagreements_outside, "inside": r.disagreements_inside}
        cf = out["circle_fit"]
        rr = nice(cf["r"] ** 2, 0.05 * cf["r"] ** 2)
        src = f"def net(x0, x1):\n    return x0*x0 + x1*x1 < {fmtp(rr)}\n"
        r2 = check_program(src, mlp, blind)
        out["snapped_circle"] = {"program": src.splitlines()[1].strip(), "snap_pass": r2.snap_pass,
                                 "outside": r2.disagreements_outside}
        src3 = (f"def net(x0, x1):\n    return (x0 - {cf['cx']})*(x0 - {cf['cx']}) + (x1 - {cf['cy']})*(x1 - {cf['cy']})"
                f" < {cf['r'] ** 2:.4f}\n")
        r3 = check_program(src3, mlp, blind)
        out["fitted_circle"] = {"snap_pass": r3.snap_pass, "outside": r3.disagreements_outside}

    # ---- ast4 today
    t1 = time.time()
    try:
        src, trace = refactor(dump_program(mlp, blind), blind, mlp, "ast4")
        out["ast4"] = {"nodes": program_size(parse_program(src, blind.arg_names))["nodes"],
                       "route": trace.get("winner"),
                       "dump_nodes": program_size(parse_program(dump_program(mlp, blind), blind.arg_names))["nodes"],
                       "s": round(time.time() - t1, 1)}
        out["ast4"]["route"] = str(out["ast4"]["route"])[:120]
        (OUT / f"{name}_{act}_ast4.py").write_text(src)
    except Exception as e:  # noqa
        out["ast4"] = {"error": repr(e)[:300]}
    (OUT / f"{name}_{act}.json").write_text(json.dumps(out, indent=1, default=float))
    print(name, act, "done", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(TASKS)
    for n in names:
        for a in ("relu", "sigmoid", "relu_wide"):
            if (OUT / f"{n}_{a}.json").exists():
                continue
            analyse(n, a)
