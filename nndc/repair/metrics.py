"""Repair metrics (efficacy, generalization, drawdown) and the special-case flag for added tests.

On one set of points, with the true labels y: the original's failures that the new version gets right
(efficacy on the repair set, generalization on the held-out set), and the original's right answers the
new version breaks (drawdown).
"""
from __future__ import annotations

import ast

import numpy as np

from ..check import program_decision
from ..compile_back import _flatten
from ..lang import evaluate, parse_program
from ..spec import Spec

MIN_SUPPORT = 5


def decide(spec: Spec, raw: np.ndarray) -> np.ndarray:
    """Decision from a net's raw output (2-D logits for argmax) or from a program's output."""
    raw = np.asarray(raw, dtype=np.float64)
    if spec.output == "argmax" and raw.ndim == 2:
        return np.argmax(raw, axis=1).astype(np.float64)
    if raw.ndim == 2:
        raw = raw[:, 0]
    return program_decision(spec, raw)


def right(spec: Spec, pred: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    if spec.output == "regression":
        return np.abs(pred - y) <= tol
    return pred == y


def repair_metrics(spec: Spec, orig: np.ndarray, new: np.ndarray, y: np.ndarray, tol: float,
                   whole_domain: bool) -> dict:
    ok_orig, ok_new = right(spec, orig, y, tol), right(spec, new, y, tol)
    fail = ~ok_orig
    fixed = float(ok_new[fail].mean()) if fail.any() else None
    m = {"efficacy": fixed, "generalization": None if whole_domain else fixed,
         "drawdown": float((~ok_new[ok_orig]).mean()) if ok_orig.any() else None,
         "accuracy": float(ok_new.mean()), "n_failing": int(fail.sum()), "n_right": int(ok_orig.sum())}
    if spec.output == "regression":
        m["max_abs_err"] = float(np.abs(new - y).max())
    return m


def _comparisons(src: str, arg_names: list[str]) -> dict[str, ast.AST]:
    """The program's comparisons with named values substituted, so each is a function of the inputs."""
    expr = _flatten(parse_program(src, arg_names).body, {})
    return {ast.unparse(n): n for n in ast.walk(expr) if isinstance(n, ast.Compare)}


def added_tests(orig_src: str, new_src: str, X: np.ndarray, arg_names: list[str]) -> list[dict]:
    old = _comparisons(orig_src, arg_names)
    rows = []
    for text, node in _comparisons(new_src, arg_names).items():
        if text in old:
            continue
        prog = parse_program(f"def net({', '.join(arg_names)}):\n    return {text}\n", arg_names)
        t = evaluate(prog, X) != 0
        support = int(min(t.sum(), (~t).sum()))
        rows.append({"test": text, "support": support, "flagged": support < MIN_SUPPORT})
    return rows
