"""P7: small boolean domains: enumerate the net's decisions and name the boolean function.

HEURISTICS: xor/seed0-1, xor/hand. Exact over the domain.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

from ..lang import evaluate, parse_program
from ..spec import Spec
from .cleanup import program_from_expr

TT_CAP = 4096


def _library(names: list[str]) -> list[str]:
    lib = ["0", "1"]
    for x in names:
        lib += [x, f"not {x}"]
    for a, b in itertools.combinations(names, 2):
        lib += [f"{a} != {b}", f"{a} == {b}", f"{a} and {b}", f"{a} or {b}",
                f"{a} and not {b}", f"not {a} and {b}", f"not ({a} or {b})", f"not ({a} and {b})",
                f"{a} or not {b}", f"not {a} or {b}"]
    if len(names) > 2:
        s = f"sum({', '.join(names)})"
        lib += [" and ".join(names), " or ".join(names)]
        lib += [f"{s} >= {k}" for k in range(1, len(names) + 1)]
        lib += [f"{s} == {k}" for k in range(0, len(names) + 1)]
    return lib


def truth_table_expr(decide: Callable[[np.ndarray], np.ndarray], spec: Spec) -> str | None:
    if spec.output != "binary" or not all(i.type == "bool" for i in spec.inputs):
        return None
    if spec.domain_size() > TT_CAP:
        return None
    X = np.array(list(itertools.product([0.0, 1.0], repeat=len(spec.inputs))))
    want = np.asarray(decide(X), dtype=float) > 0.5
    for expr in _library(spec.arg_names):
        got = evaluate(parse_program(program_from_expr(expr, spec), spec.arg_names), X) > spec.threshold
        if np.array_equal(got, want):
            return expr
    return None
