"""P9: emission of readable code: factored sums, ordered terms, named derived variables.

HEURISTICS: sparse8/hand, sum8/hand.
"""
from __future__ import annotations

from ..spec import Spec
from ..symnet import Affine, fmt

SUM_MIN_VARS = 3


def join_terms(items: list[tuple[float, str | None]]) -> str:
    """Sum of coef * atom (atom None = constant term) in the given order; '0' when empty."""
    out: list[str] = []
    for coef, atom in items:
        if coef == 0:
            continue
        if atom is None:
            body = fmt(abs(coef))
        else:
            body = atom if abs(coef) == 1 else f"{fmt(abs(coef))} * {atom}"
        if not out:
            out.append(f"-{body}" if coef < 0 else body)
        else:
            out.append(f" {'-' if coef < 0 else '+'} {body}")
    return "".join(out) or "0"


def var_items(a: Affine) -> list[tuple[float, str]]:
    """Variable terms of an affine form; three or more equal coefficients become c * sum(...)."""
    groups: dict[float, list[str]] = {}
    for v, c in a.coef.items():
        groups.setdefault(c, []).append(v)
    items: list[tuple[float, str]] = []
    for c, vs in groups.items():
        if len(vs) >= SUM_MIN_VARS:
            items.append((c, f"sum({', '.join(vs)})"))
        else:
            items += [(c, v) for v in vs]
    return items


def affine_src(a: Affine) -> str:
    return join_terms(var_items(a) + [(a.const, None)])


def program_from_expr(expr: str, spec: Spec, derived: dict[str, str] | None = None) -> str:
    lines = [f"def net({', '.join(spec.arg_names)}):"]
    for name, value in (derived or {}).items():
        lines.append(f"    {name} = {value}")
    lines.append(f"    return {expr}")
    return "\n".join(lines) + "\n"
