"""The AST heuristics tool: build candidates along several routes, keep the best that passes the checker.

Deterministic, no LLM. It sees only the dump, the blind spec and the net (through the checker),
never the task name or the target.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .check import check_program, net_output
from .heuristics.boundary import boundary_expr
from .heuristics.boundary_scan import scan_boundary_expr
from .heuristics.bounds import prune_dead_and_linear
from .heuristics.cleanup import program_from_expr
from .heuristics.decision_list import decision_list
from .heuristics.sigmoid import knot_fit, linear_cascade, linear_scores, merge_steps, replace_units
from .heuristics.fit import fit_expr
from .heuristics.kinks import kink_merge
from .heuristics.merge import merge_units, normalize
from .heuristics.motifs import as_relu_sum, motif_expr
from .heuristics.snap import snap_net
from .heuristics.spline import spline_expr
from .heuristics.truthtable import truth_table_expr
from .lang import ProgramError, parse_program, program_size
from .model import MLP
from .spec import Spec
from .symnet import SymNet, from_dump, input_bounds, to_program

ROUTES = ["structural-exact", "structural-snap", "spline", "boundary", "boundary-scan", "truth-table", "fit",
          "decision-list-snap", "decision-list",   # snapped first: when neither is exact, prefer round constants
          "sigmoid-replace-snap", "sigmoid-replace", "linear-cascade", "linear-scores", "knot-fit",   # non-ReLU nets
          "kink-merge"]


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


def scan_directions(net: SymNet, spec: Spec, top: int = 4) -> list[dict[str, float]]:
    """Directions of the most influential first-layer units, then the input axes."""
    inputs = set(net.inputs)
    post = input_bounds(spec)
    scored = []
    for u in net.units:
        if u.act == "relu" and u.pre.coef and set(u.pre.coef) <= inputs:
            lo, hi = u.pre.interval(post)
            weight = sum(abs(c.get(u.name, 0.0)) for c in [x.pre.coef for x in net.units] + [o.coef for o in net.outputs])
            scored.append((weight * max(hi, 0.0), dict(u.pre.coef)))
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:top]] + [{v: 1.0} for v in net.inputs]


def refactor(dump_src: str, spec: Spec, mlp: MLP) -> tuple[str, dict]:
    if spec.task:
        raise ValueError("refactor needs a blind spec (the task name must not be visible)")
    lines: list[str] = []
    cands: list[Candidate] = []
    extra: dict = {}

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
        # spline does its own snapping of kinks and slopes, so try it on the exact net as well as the snapped one
        for spline_net in (n2, n6):
            rs_s = as_relu_sum(spline_net)
            add("spline", lambda rs_s=rs_s: (None if rs_s is None else
                                             (lambda e: None if e is None else program_from_expr(e, spec))(
                                                 spline_expr(rs_s, spec))),
                ["P1", "P2", "P3", "P5", "P9"])

    if spec.output == "binary":
        def boundary() -> str | None:
            r = boundary_expr(n2, spec)
            return None if r is None else program_from_expr(r[0], spec, r[1])
        add("boundary", boundary, ["P1", "P2", "P6", "P9"])

        def boundary_scan() -> str | None:
            r = scan_boundary_expr(lambda X: net_output(mlp, spec, X)[1], spec, scan_directions(n2, spec))
            return None if r is None else program_from_expr(r[0], spec, r[1])
        add("boundary-scan", boundary_scan, ["P1", "P2", "P6b", "P9"])

        def truth() -> str | None:
            e = truth_table_expr(lambda X: net_output(mlp, spec, X)[0], spec)
            return None if e is None else program_from_expr(e, spec)
        add("truth-table", truth, ["P7"])

    if spec.output == "regression":
        def fit() -> str | None:
            e = fit_expr(lambda X: mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64), spec)
            return None if e is None else program_from_expr(e, spec)
        add("fit", fit, ["P8"])

    if spec.output == "argmax":
        dl = decision_list(n2, spec, mlp)
        if dl is None:
            lines.append("decision-list: not applicable")
        else:
            lines += dl[1]
            for route, src in dl[0]:
                passes = ["P1", "P2", "P10", "P9"] if route == "decision-list" else ["P1", "P2", "P10", "P3", "P9"]
                add(route, lambda src=src: src, passes)

    if any(u.act not in ("relu", "linear") for u in net0.units):   # sigmoid moves; ReLU nets never reach them
        r1, tr = replace_units(n1, spec, mlp)
        m1, tm = merge_steps(r1, spec, mlp)
        p1, tp = prune_dead_and_linear(m1, spec)
        lines += [f"sigmoid: {x}" for x in tr + tm + tp]
        add("sigmoid-replace", lambda: _emit(p1, spec)[0], ["S1", "S2"])
        s1, ts = snap_net(p1, spec)
        s2, _ = prune_dead_and_linear(s1, spec)
        lines += [f"sigmoid snap: {x}" for x in ts]
        add("sigmoid-replace-snap", lambda: _emit(s2, spec)[0], ["S1", "S2", "P3"])
        if spec.output == "argmax":
            dl = decision_list(p1, spec, mlp)
            if dl is None:
                lines.append("sigmoid decision-list: not applicable")
            else:
                lines += [f"sigmoid {x}" for x in dl[1]]
                for route, src in dl[0]:
                    add(route, lambda src=src: src, ["S1", "S2", "P10", "P9"])
            add("linear-scores", lambda: linear_scores(spec, mlp), ["S3"])
            add("linear-cascade", lambda: linear_cascade(spec, mlp), ["S4"])
        if spec.output == "regression":
            add("knot-fit", lambda: knot_fit(spec, mlp), ["S5"])
        src_k, extra["kink_merge"] = kink_merge(spec, mlp)
        add("kink-merge", lambda: src_k, ["K1"])

    passing = [c for c in cands if c.check["snap_pass"]]
    if passing:
        # simplest passing program first (the structural-exact route always passes exact, so ranking
        # exactness first would never let a snapped program win); exactness breaks ties
        best = min(passing, key=lambda c: (c.nodes, not c.check["exact_pass"], ROUTES.index(c.route)))
    else:
        best = next((c for c in cands if c.route == "structural-exact"), None)
        if best is None:
            best = Candidate("dump", dump_src, [], check_program(dump_src, mlp, spec).to_json(), None)
    trace = {"candidates": [asdict(c) for c in cands], "winner": best.route, "trace_lines": lines} | extra
    return best.src, trace


def run_ast(rd: Path, name: str = "ast") -> dict:
    """Refactor a run's dump.py with the tool (blind spec only); write <name>.py + <name>_trace.json; re-evaluate.
    A new tool version gets a new name (ast2), so the older version's files stay."""
    from .run import evaluate_run, load_run, record_git

    spec, mlp = load_run(rd)
    blind = Spec.from_json(spec.blind())
    src, trace = refactor((rd / "dump.py").read_text(encoding="utf-8"), blind, mlp)
    (rd / f"{name}.py").write_text(f"# produced by nndc ast (route: {trace['winner']})\n{src}", encoding="utf-8")
    (rd / f"{name}_trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    record_git(rd, name)
    return evaluate_run(rd)
