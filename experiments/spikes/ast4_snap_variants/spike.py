"""Spike: kink-merge snap variants. Read-only on the repo (loads run folders, writes nothing there).

Variants of snap_fit: window new (line range) / old (max span), constant rounded / raw.
For each run: per (m, off, tol, variant) -> pass, nodes. Then compare route policies.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from nndc.check import check_program  # noqa: E402
from nndc.heuristics import kinks as K  # noqa: E402
from nndc.heuristics.decision_list import nicest_in  # noqa: E402
from nndc.heuristics.snap import nice  # noqa: E402
from nndc.lang import parse_program, program_size  # noqa: E402
from nndc.run import load_run  # noqa: E402


def snap_var(fit, spec, tol, U, y, window, const_round):
    lo, span = K.unit_box(spec)
    names = spec.arg_names
    X = lo + U * span
    box = lo + K._corners(len(names)) * span
    geo = {}
    for n, t, _ in fit.terms:
        big = max(n, key=lambda v: abs(n[v]))
        n2 = {v: (c if v == big else nice(c, tol)) for v, c in n.items()}
        n2 = {v: c for v, c in n2.items() if c != 0}
        if window == "new":
            width = sum(abs(n2.get(v, 0.0)) * float(s) for v, s in zip(names, span))
        else:
            width = float(np.max(span))
        t2 = nicest_in(t - tol * width, t + tol * width)
        z = box @ np.array([n2.get(v, 0.0) for v in names]) - t2
        if z.min() >= 0 or z.max() <= 0:
            continue
        geo[(tuple(sorted(n2.items())), t2)] = (n2, t2)
    lines = list(geo.values())
    H = K._hinges(X, names, lines)
    c, _ = K.minimax(np.column_stack([np.ones(len(X)), X] + H), y)
    k = len(names)
    linear = {names[i]: K._nicest_rel(float(c[1 + i]), tol) for i in range(k)}
    strengths = [K._nicest_rel(float(s), tol) for s in c[1 + k:]]
    rest = y - X @ np.array([linear[v] for v in names]) - sum((s * h for s, h in zip(strengths, H)), np.zeros(len(X)))
    const = float((rest.max() + rest.min()) / 2)
    if tol > 0 and const_round:
        r = float(y.max() - y.min())
        const = nicest_in(const - tol * r, const + tol * r)
    return K.RealFit(const, linear, [(n, t, s) for (n, t), s in zip(lines, strengths)])


VARIANTS = {"ast3": ("new", True), "rawconst": ("new", False), "oldwin": ("old", True), "both": ("old", False)}


def grid(rd: Path):
    spec, mlp = load_run(rd)
    if K.applies(spec, mlp) is not None:
        return None
    lo, span = K.unit_box(spec)
    d = len(lo)
    rng = np.random.default_rng(K.FIT_SEED)
    U = np.vstack([rng.uniform(0.0, 1.0, (K.FIT_SAMPLES, d)), K._corners(d)])
    y = mlp.forward((lo + U * span).astype(np.float32))[:, 0].astype(np.float64)
    kinks = K.expand(spec, mlp).kinks
    line_sets = {off: K.merge_lines(kinks, off) for off in K.MERGE_OFFS}
    rows = []
    for m in range(1, K.MAX_LINES + 1):
        for off in K.MERGE_OFFS:
            if len(line_sets[off]) < m:
                continue
            lines, coefs, _ = K.refine(line_sets[off][:m], U, y)
            fit = K.to_real(lines, coefs, spec)
            for tol in K.SNAP_LEVELS:
                for name, (w, cr) in VARIANTS.items():
                    f = snap_var(fit, spec, tol, U, y, w, cr)
                    src = K.fit_program(f, spec)
                    r = check_program(src, mlp, spec)
                    nodes = program_size(parse_program(src, spec.arg_names))["nodes"]
                    rows.append({"m": m, "off": off, "tol": tol, "var": name, "pass": bool(r.snap_pass),
                                 "nodes": nodes, "src": src})
    return rows


if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    for rel in sys.argv[1:]:
        rows = grid(REPO / rel)
        (out / (rel.replace("/", "_") + ".json")).write_text(json.dumps(rows))
        print(rel, "n/a" if rows is None else len(rows), flush=True)
