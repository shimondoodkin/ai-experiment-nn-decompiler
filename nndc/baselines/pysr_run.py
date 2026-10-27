"""B1/B2: PySR symbolic regression on samples of the net, raw (its own pick) and with this repo's snapping.

B1 is PySR's `get_best()`. B2 snaps the constants of every Pareto-front equation with `nice()` and keeps
the smallest that passes the checker. Binary nets are fitted on the score out - threshold; argmax nets on
their logits (one equation per output, combined with `argmax`).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import sympy

from ..heuristics.snap import nice
from ..lang import ProgramError, parse_program
from ..run import load_run, record_git
from ..spec import Spec
from .common import net_samples, smallest_passing, write_info

N_PYSR = 5000
PYSR_SETTINGS = dict(binary_operators=["+", "-", "*", "max", "min", "greater", "cond"],
                     unary_operators=["abs", "relu"], maxsize=25, niterations=40, timeout_in_seconds=120,
                     deterministic=True, parallelism="serial", random_state=0)
COEF_REL, ADD_FRAC, CMP_FRAC = 0.05, 0.01, 0.01


class UnsupportedExpr(ValueError):
    pass


def _num(v) -> str:
    return repr(float(v))


def _src(e) -> str:
    if isinstance(e, sympy.Symbol):
        return e.name
    if e is sympy.true:
        return "True"
    if e is sympy.false:
        return "False"
    if e.is_Number:
        return _num(e)
    if isinstance(e, sympy.Add):
        return "(" + " + ".join(_src(a) for a in e.args) + ")"
    if isinstance(e, sympy.Mul):
        num, den = [], []
        for a in e.args:
            if isinstance(a, sympy.Pow) and a.exp.is_Number and a.exp < 0:
                if not a.base.is_Number:
                    raise UnsupportedExpr(f"division by a non-constant: {e}")
                num.append(sympy.Pow(a.base, a.exp))
            else:
                num.append(a)
        return "(" + " * ".join(_src(a) for a in num + den) + ")"
    if isinstance(e, sympy.Pow):
        if e.base.is_Number and e.exp.is_Number:
            return _num(e)
        if e.exp.is_Integer and e.exp > 0:
            return "(" + " * ".join([_src(e.base)] * int(e.exp)) + ")"
        raise UnsupportedExpr(f"power {e}")
    if isinstance(e, sympy.Abs):
        return f"abs({_src(e.args[0])})"
    if isinstance(e, (sympy.Max, sympy.Min)):
        return f"{'max' if isinstance(e, sympy.Max) else 'min'}({', '.join(_src(a) for a in e.args)})"
    if isinstance(e, sympy.Piecewise):
        (val, cond), rest = e.args[0], e.args[1:]
        if cond is sympy.true:
            return _src(val)
        other = _src(sympy.Piecewise(*rest)) if rest else "0.0"
        return f"({_src(val)} if {_src(cond)} else {other})"
    if isinstance(e, sympy.core.relational.Relational):
        op = {sympy.StrictGreaterThan: ">", sympy.GreaterThan: ">=", sympy.StrictLessThan: "<",
              sympy.LessThan: "<=", sympy.Eq: "==", sympy.Ne: "!="}.get(type(e))
        if op is None:
            raise UnsupportedExpr(f"relation {e}")
        return f"({_src(e.lhs)} {op} {_src(e.rhs)})"
    if isinstance(e, (sympy.And, sympy.Or)):
        return "(" + (" and " if isinstance(e, sympy.And) else " or ").join(_src(a) for a in e.args) + ")"
    if isinstance(e, sympy.Not):
        return f"(not {_src(e.args[0])})"
    raise UnsupportedExpr(f"unsupported {type(e).__name__}: {e}")


def sympy_to_program(expr, spec: Spec) -> str:
    """One expression as a program; raises UnsupportedExpr outside the language."""
    body = _src(expr)
    src = f"def net({', '.join(spec.arg_names)}):\n    return {body}\n"
    try:
        parse_program(src, spec.arg_names)
    except ProgramError as e:
        raise UnsupportedExpr(str(e)) from None
    return src


def _snapped(v: float, tol: float):
    s = nice(float(v), tol)
    return sympy.Integer(int(s)) if float(s).is_integer() else sympy.Float(s)


def snap_expr(expr, out_range: float, in_range: float):
    """Snap each Float by where it sits: a Mul factor 5% relative, a comparison argument 1% of the
    input range, anything else (Add terms, max/min/piecewise values) 1% of the output range."""
    def rec(e, ctx: str):
        if isinstance(e, sympy.Float):
            v = float(e)
            tol = COEF_REL * abs(v) if ctx == "mul" else CMP_FRAC * in_range if ctx == "cmp" else ADD_FRAC * out_range
            return _snapped(v, tol)
        if e.is_Atom:
            return e
        child = "mul" if isinstance(e, sympy.Mul) else "cmp" if isinstance(e, sympy.core.relational.Relational) \
            else "add"
        return e.func(*[rec(a, child) for a in e.args])
    return rec(expr, "add")


def _program(exprs: list, spec: Spec) -> str:
    if spec.output != "argmax":
        return sympy_to_program(exprs[0], spec)
    body = "argmax(" + ", ".join(_src(e) for e in exprs) + ")"
    src = f"def net({', '.join(spec.arg_names)}):\n    return {body}\n"
    try:
        parse_program(src, spec.arg_names)
    except ProgramError as e:
        raise UnsupportedExpr(str(e)) from None
    return src


def run_pysr(rd: Path) -> dict:
    from pysr import PySRRegressor

    t0 = time.perf_counter()
    spec_full, mlp = load_run(rd)
    spec = Spec.from_json(spec_full.blind())
    X, raw, _ = net_samples(spec, mlp)
    X, y = X[:N_PYSR], raw[:N_PYSR]
    with tempfile.TemporaryDirectory() as tmp:
        model = PySRRegressor(**PYSR_SETTINGS, progress=False, verbosity=0, output_directory=tmp)
        model.fit(X, y, variable_names=spec.arg_names)
        fronts = model.equations_ if isinstance(model.equations_, list) else [model.equations_]
        best = model.sympy()
    best = best if isinstance(best, list) else [best]
    info = {"settings": PYSR_SETTINGS, "n_samples": int(len(X)), "best": [str(b) for b in best],
            "fronts": [[{"complexity": int(r.complexity), "loss": float(r.loss), "equation": str(r.equation)}
                        for r in f.itertuples()] for f in fronts], "errors": []}
    rd = Path(rd)
    try:
        (rd / "pysr.py").write_text(_program(best, spec), encoding="utf-8")
    except UnsupportedExpr as e:
        info["errors"].append(f"pysr: {e}")
        (rd / "pysr.py").unlink(missing_ok=True)

    yr = np.atleast_2d(y.T).T
    out_ranges = [float(yr[:, k].max() - yr[:, k].min()) or 1.0 for k in range(yr.shape[1])]
    in_range = max(i.hi - i.lo for i in spec.inputs)
    snapped_fronts = [[snap_expr(e, out_ranges[k], in_range) for e in f["sympy_format"]]
                      for k, f in enumerate(fronts)]
    complexities = [list(f["complexity"]) for f in fronts]
    combos = []   # one candidate per complexity cap: each output's best equation within the cap
    for cap in sorted({c for cs in complexities for c in cs}):
        pick = []
        for k, cs in enumerate(complexities):
            ok = [i for i, c in enumerate(cs) if c <= cap]
            if not ok:
                break
            pick.append(snapped_fronts[k][ok[-1]])
        if len(pick) == len(fronts):
            combos.append(pick)
    cands = []
    for exprs in combos:
        try:
            cands.append(_program(exprs, spec))
        except UnsupportedExpr as e:
            info["errors"].append(f"pysr_snap: {e}")
    idx, rows = smallest_passing(cands, mlp, spec) if cands else (None, [])
    info["snap_candidates"] = rows
    info["snap_passed"] = idx is not None
    if cands:
        # nothing passes: report the snapped largest candidate as a failing program, as the tree does
        (rd / "pysr_snap.py").write_text(cands[idx if idx is not None else -1], encoding="utf-8")
    else:
        (rd / "pysr_snap.py").unlink(missing_ok=True)
    info["seconds"] = round(time.perf_counter() - t0, 2)
    write_info(rd, "pysr", info)
    record_git(rd, "pysr")
    return info
