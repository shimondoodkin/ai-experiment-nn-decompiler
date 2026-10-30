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
PYSR_IRR_SETTINGS = {**PYSR_SETTINGS, "binary_operators": PYSR_SETTINGS["binary_operators"] + ["/"],
                     "unary_operators": PYSR_SETTINGS["unary_operators"] + ["exp", "log", "sqrt"]}   # irrational runs
COEF_REL, ADD_FRAC, CMP_FRAC = 0.05, 0.01, 0.01


class UnsupportedExpr(ValueError):
    pass


def _num(v) -> str:
    return repr(float(v))


def _src(e, irr: bool = False) -> str:
    """Program text of a sympy expression. irr=True (pysr_irr) also prints exp, log, sqrt and division by a
    non-constant, which the language allows since the irrational-functions project."""
    def rec(a) -> str:
        return _src(a, irr)
    if isinstance(e, sympy.Symbol):
        return e.name
    if e is sympy.true:
        return "True"
    if e is sympy.false:
        return "False"
    if e.is_Number:
        return _num(e)
    if isinstance(e, sympy.Add):
        return "(" + " + ".join(rec(a) for a in e.args) + ")"
    if isinstance(e, sympy.Mul):
        num, den = [], []
        for a in e.args:
            if isinstance(a, sympy.Pow) and a.exp.is_Number and a.exp < 0:
                if not a.base.is_Number:
                    if not irr:
                        raise UnsupportedExpr(f"division by a non-constant: {e}")
                    den.append(sympy.Pow(a.base, -a.exp))
                    continue
                num.append(sympy.Pow(a.base, a.exp))
            else:
                num.append(a)
        body = " * ".join(rec(a) for a in num) or "1.0"
        if den:
            return f"({body} / ({' * '.join(rec(a) for a in den)}))"
        return f"({body})"
    if isinstance(e, sympy.Pow):
        if e.base.is_Number and e.exp.is_Number:
            return _num(e)
        if e.exp.is_Integer and e.exp > 0:
            return "(" + " * ".join([rec(e.base)] * int(e.exp)) + ")"
        if irr and e.exp == sympy.Rational(1, 2):
            return f"sqrt({rec(e.base)})"
        if irr and e.exp.is_Number and e.exp < 0:
            return f"(1.0 / {rec(sympy.Pow(e.base, -e.exp))})"
        raise UnsupportedExpr(f"power {e}")
    if irr and isinstance(e, (sympy.exp, sympy.log)):
        return f"{'exp' if isinstance(e, sympy.exp) else 'log'}({rec(e.args[0])})"
    if isinstance(e, sympy.Abs):
        return f"abs({rec(e.args[0])})"
    if isinstance(e, (sympy.Max, sympy.Min)):
        return f"{'max' if isinstance(e, sympy.Max) else 'min'}({', '.join(rec(a) for a in e.args)})"
    if isinstance(e, sympy.Piecewise):
        (val, cond), rest = e.args[0], e.args[1:]
        if cond is sympy.true:
            return rec(val)
        other = rec(sympy.Piecewise(*rest)) if rest else "0.0"
        return f"({rec(val)} if {rec(cond)} else {other})"
    if isinstance(e, sympy.core.relational.Relational):
        op = {sympy.StrictGreaterThan: ">", sympy.GreaterThan: ">=", sympy.StrictLessThan: "<",
              sympy.LessThan: "<=", sympy.Eq: "==", sympy.Ne: "!="}.get(type(e))
        if op is None:
            raise UnsupportedExpr(f"relation {e}")
        return f"({rec(e.lhs)} {op} {rec(e.rhs)})"
    if isinstance(e, (sympy.And, sympy.Or)):
        return "(" + (" and " if isinstance(e, sympy.And) else " or ").join(rec(a) for a in e.args) + ")"
    if isinstance(e, sympy.Not):
        return f"(not {rec(e.args[0])})"
    raise UnsupportedExpr(f"unsupported {type(e).__name__}: {e}")


def sympy_to_program(expr, spec: Spec, irr: bool = False) -> str:
    """One expression as a program; raises UnsupportedExpr outside the language."""
    body = _src(expr, irr)
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


def _program(exprs: list, spec: Spec, irr: bool = False) -> str:
    if spec.output != "argmax":
        return sympy_to_program(exprs[0], spec, irr)
    body = "argmax(" + ", ".join(_src(e, irr) for e in exprs) + ")"
    src = f"def net({', '.join(spec.arg_names)}):\n    return {body}\n"
    try:
        parse_program(src, spec.arg_names)
    except ProgramError as e:
        raise UnsupportedExpr(str(e)) from None
    return src


def run_pysr(rd: Path, name: str = "pysr", settings: dict = PYSR_SETTINGS, irr: bool = False) -> dict:
    """Writes <name>.py, <name>_snap.py and <name>_info.json."""
    from pysr import PySRRegressor

    t0 = time.perf_counter()
    spec_full, mlp = load_run(rd)
    spec = Spec.from_json(spec_full.blind())
    X, raw, _ = net_samples(spec, mlp)
    X, y = X[:N_PYSR], raw[:N_PYSR]
    with tempfile.TemporaryDirectory() as tmp:
        model = PySRRegressor(**settings, progress=False, verbosity=0, output_directory=tmp)
        model.fit(X, y, variable_names=spec.arg_names)
        fronts = model.equations_ if isinstance(model.equations_, list) else [model.equations_]
        best = model.sympy()
    best = best if isinstance(best, list) else [best]
    info = {"settings": settings, "n_samples": int(len(X)), "best": [str(b) for b in best],
            "fronts": [[{"complexity": int(r.complexity), "loss": float(r.loss), "equation": str(r.equation)}
                        for r in f.itertuples()] for f in fronts], "errors": []}
    rd = Path(rd)
    try:
        (rd / f"{name}.py").write_text(_program(best, spec, irr), encoding="utf-8")
    except UnsupportedExpr as e:
        info["errors"].append(f"{name}: {e}")
        (rd / f"{name}.py").unlink(missing_ok=True)

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
            cands.append(_program(exprs, spec, irr))
        except UnsupportedExpr as e:
            info["errors"].append(f"{name}_snap: {e}")
    idx, rows = smallest_passing(cands, mlp, spec) if cands else (None, [])
    info["snap_candidates"] = rows
    info["snap_passed"] = idx is not None
    if cands:
        # nothing passes: report the snapped largest candidate as a failing program, as the tree does
        (rd / f"{name}_snap.py").write_text(cands[idx if idx is not None else -1], encoding="utf-8")
    else:
        (rd / f"{name}_snap.py").unlink(missing_ok=True)
    info["seconds"] = round(time.perf_counter() - t0, 2)
    write_info(rd, name, info)
    record_git(rd, name)
    return info


def run_pysr_irr(rd: Path) -> dict:
    """PySR with exp, log, sqrt and / added (irrational runs); writes pysr_irr.py and pysr_irr_snap.py."""
    return run_pysr(rd, "pysr_irr", PYSR_IRR_SETTINGS, irr=True)
