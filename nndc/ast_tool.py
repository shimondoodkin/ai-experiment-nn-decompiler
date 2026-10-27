"""The AST heuristics tool: build candidates along several routes, keep the best that passes the checker.

Deterministic, no LLM. It sees only the dump, the blind spec and the net (through the checker),
never the task name or the target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .check import check_program, net_output
from .heuristics.boundary import boundary_expr
from .heuristics.bounds import prune_dead_and_linear
from .heuristics.cleanup import program_from_expr
from .heuristics.fit import fit_expr
from .heuristics.merge import merge_units, normalize
from .heuristics.motifs import as_relu_sum, motif_expr
from .heuristics.snap import snap_net
from .heuristics.spline import spline_expr
from .heuristics.truthtable import truth_table_expr
from .lang import ProgramError, parse_program, program_size
from .model import MLP
from .spec import Spec
from .symnet import SymNet, from_dump, to_program

ROUTES = ["structural-exact", "structural-snap", "spline", "boundary", "truth-table", "fit"]


@dataclass
class Candidate:
    route: str
    src: str
    passes: list[str]
    check: dict
    nodes: int | None


def _emit(net: SymNet, spec: Spec) -> tuple[str, bool]:
    """Program for a (pruned/merged) net: motif-rewritten when it is one hidden layer, else plain."""
    rs = as_relu_sum(net) if spec.output != "argmax" else None
    if rs is not None:
        return program_from_expr(motif_expr(rs), spec), True
    return to_program(net), False


def refactor(dump_src: str, spec: Spec, mlp: MLP) -> tuple[str, dict]:
    if spec.task:
        raise ValueError("refactor needs a blind spec (the task name must not be visible)")
    lines: list[str] = []
    cands: list[Candidate] = []

    def add(route: str, build: Callable[[], str | None], passes: list[str]) -> None:
        try:
            src = build()
        except (ValueError, ProgramError, ZeroDivisionError) as e:
            lines.append(f"{route}: not applicable ({e})")
            return
        if src is None:
            lines.append(f"{route}: not applicable")
            return
        try:
            nodes = program_size(parse_program(src, spec.arg_names))["nodes"]
        except ProgramError as e:
            lines.append(f"{route}: invalid program ({e})")
            return
        r = check_program(src, mlp, spec)
        cands.append(Candidate(route, src, passes, r.to_json(), nodes))

    net0 = from_dump(dump_src, spec)
    n1, t1 = prune_dead_and_linear(net0, spec)
    n2, t2 = merge_units(normalize(n1), spec, near=False)
    lines += t1 + t2

    add("structural-exact", lambda: _emit(n2, spec)[0], ["P1", "P2", "P4", "P9"])

    n3, t3 = merge_units(normalize(n1), spec, near=True)
    n4, t4 = snap_net(n3, spec)
    n5, t5 = prune_dead_and_linear(n4, spec)
    n6, t6 = merge_units(normalize(n5), spec, near=False)
    lines += [f"snap route: {x}" for x in t3 + t4 + t5 + t6]
    add("structural-snap", lambda: _emit(n6, spec)[0], ["P1", "P2", "P3", "P4", "P9"])

    if spec.output != "argmax":
        rs6 = as_relu_sum(n6)
        add("spline", lambda: (None if rs6 is None else
                               (lambda e: None if e is None else program_from_expr(e, spec))(spline_expr(rs6, spec))),
            ["P1", "P2", "P3", "P5", "P9"])

    if spec.output == "binary":
        def boundary() -> str | None:
            r = boundary_expr(n2, spec)
            return None if r is None else program_from_expr(r[0], spec, r[1])
        add("boundary", boundary, ["P1", "P2", "P6", "P9"])

        def truth() -> str | None:
            e = truth_table_expr(lambda X: net_output(mlp, spec, X)[0], spec)
            return None if e is None else program_from_expr(e, spec)
        add("truth-table", truth, ["P7"])

    if spec.output == "regression":
        def fit() -> str | None:
            e = fit_expr(lambda X: mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64), spec)
            return None if e is None else program_from_expr(e, spec)
        add("fit", fit, ["P8"])

    passing = [c for c in cands if c.check["snap_pass"]]
    if passing:
        # simplest passing program first (the structural-exact route always passes exact, so ranking
        # exactness first would never let a snapped program win); exactness breaks ties
        best = min(passing, key=lambda c: (c.nodes, not c.check["exact_pass"], ROUTES.index(c.route)))
    else:
        best = next((c for c in cands if c.route == "structural-exact"), None)
        if best is None:
            best = Candidate("dump", dump_src, [], check_program(dump_src, mlp, spec).to_json(), None)
    trace = {"candidates": [asdict(c) for c in cands], "winner": best.route, "trace_lines": lines}
    return best.src, trace
