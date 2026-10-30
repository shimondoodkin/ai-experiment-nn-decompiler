"""runs_irr/REPORT.md: the irrational-function nets (design docs/design/2026-10-30-irrational-functions-design.md).

Reads only <base>/<variant>/<task>/<seed>/irr_analysis.json (written by `nndc irr-analyse`), each variant's
heldout.json and <base>/before_minimax.json (the ast5 results before the gate-view design, for its third prediction);
it runs no checks.
"""
from __future__ import annotations

import json
from pathlib import Path

VARIANTS = ("relu", "relu_wide", "sigmoid", "tanh")
DEV_SEEDS = ("seed0", "seed1", "seed2")
NEAR_WITHIN = 0.03


def load_rows(base: Path) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        root = Path(base) / variant
        hp = root / "heldout.json"
        held = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else None
        names = set(held["runs"]) if held else set()
        for p in sorted(root.glob(f"*/*/irr_analysis.json")):
            a = json.loads(p.read_text(encoding="utf-8"))
            a["variant"] = variant
            a["heldout"] = f"{p.parent.parent.name}/{p.parent.name}" in names
            a["freeze"] = held.get("freeze") if held else None
            rows.append(a)
    return rows


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({round(100 * k / n)}%)" if n else "–"


def _cnt(k: int, n: int) -> str:
    return f"{k} ({round(100 * k / n)}%)" if n else "0"


def _yn(b) -> str:
    return "–" if b is None else "yes" if b else "no"


def _is_regression(a: dict) -> bool:
    return a.get("family") != "conic"


def _near_ok(a: dict) -> bool:
    """Found, or the near formula is of the task's family and within 3% of the output range."""
    return bool(a["formula_found"] or (a.get("near_family_is_task_family") and a.get("near_err") is not None
                                       and a["near_err"] <= NEAR_WITHIN))


def _results_md(rows: list[dict]) -> str:
    out = ["## Results", "",
           "One row per net. *formula found*: a passing `ast5` family program of the task's family with at most 4 "
           "correction terms (the number in brackets). *won*: that program has fewer nodes than both the hinge-only "
           "fit (1-D) and `ast4`. *near formula*: where no formula passes, `ast5`'s near formula (family, max error as "
           "a share of the output range; for a conic, the share of inputs decided differently). *explain*: the run's "
           "`explain.md` (formula, residual and gate view).", "",
           "| variant | task | seed | learned | true vs net | formula.py | formula found | route pick | ast4 nodes "
           "| ast5 nodes (route) | near formula | hinge-only | won | PySR | explain |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in rows:
        found = f"yes ({a['found_terms']})" if a["formula_found"] else "no"
        h = a.get("hinge_only") or {}
        hinge = f"{h['hinges']} hinges, {h['nodes']} nodes" if h.get("passed") else ("fails" if h else "–")
        a4 = f"{a['ast4_nodes']}" + ("" if a["ast4_pass"] else " (fails)") if a["ast4_nodes"] is not None else "–"
        a5 = (f"{a['ast5_nodes']} ({a['ast5_route']})" + ("" if a["ast5_pass"] else " (fails)")
              if a["ast5_nodes"] is not None else "–")
        tvn = "–" if a.get("true_vs_net") is None else f"{100 * a['true_vs_net']:.1f}%"
        seed = a["seed"] + (" (held-out)" if a["heldout"] else "")
        near = (f"{a['near_family']} {100 * a['near_err']:.1f}%"
                if not a["formula_found"] and a.get("near_family") and a.get("near_err") is not None else "–")
        link = f"[explain]({a['variant']}/{a['task']}/{a['seed']}/explain.md)" if a.get("explain") else "–"
        out.append(f"| {a['variant']} | {a['task']} | {seed} | {_yn(a['learned'])} | {tvn} | "
                   f"{_yn(a.get('formula_py_pass'))} | {found} | {a.get('route_pick') or '–'} | {a4} | {a5} | {near} | "
                   f"{hinge} | {_yn(a['won_on_size'])} | {_yn(a.get('pysr_found'))} | {link} |")
    return "\n".join(out) + "\n"


def _model_md(rows: list[dict], base: Path) -> str:
    out = ["## How the nets model the functions", "",
           "1-D nets: kinks (ReLU) or unit centres (sigmoid, tanh) inside the box, stacked units counted once, against "
           "the curvature-optimal knots for the same count (density √|f″|). *gap*: mean distance between the sorted "
           "lists, as a share of the box. *max error at*: where the net is furthest from the true function.", "",
           "| variant | task | seed | units in box | dead | gap | max error at | true vs net |",
           "|---|---|---|---|---|---|---|---|"]
    one = [a for a in rows if "knot_gap" in a["model"]]
    for a in one:
        m = a["model"]
        pts = m.get("kinks") if m.get("kinks") is not None else m.get("centres") or []
        dead = "–" if m.get("dead_units") is None else m["dead_units"]
        gap = "–" if m["knot_gap"] is None else f"{100 * m['knot_gap']:.1f}%"
        tvn = "–" if a.get("true_vs_net") is None else f"{100 * a['true_vs_net']:.1f}%"
        out.append(f"| {a['variant']} | {a['task']} | {a['seed']} | {len(pts)} | {dead} | {gap} | {m['max_err_at']} "
                   f"| {tvn} |")
    circ = [a for a in rows if "pieces" in a["model"]]
    if circ:
        out += ["", "Binary 2-D nets: the decision boundary along 720 rays from the true centre. *pieces*: distinct "
                "ReLU activation patterns on it (straight sides).", "",
                "| variant | task | seed | pieces | radius min–max | snapped conic |", "|---|---|---|---|---|---|"]
        for a in circ:
            m = a["model"]
            r = m.get("radius_min_max")
            out.append(f"| {a['variant']} | {a['task']} | {a['seed']} | {m['pieces'] if m['pieces'] is not None else '–'} | "
                       f"{'–' if not r else f'{r[0]}–{r[1]}'} | {m.get('conic') or '–'} |")
    reg = [a for a in rows if "err_by_radius" in a["model"]]
    if reg:
        out += ["", "2-D regression nets: the largest |net − true| in 10 rings of distance from the centre, innermost "
                "first.", "", "| variant | task | seed | error by ring |", "|---|---|---|---|"]
        for a in reg:
            ring = ", ".join("–" if e is None else f"{e:.3g}" for e in a["model"]["err_by_radius"])
            out.append(f"| {a['variant']} | {a['task']} | {a['seed']} | {ring} |")
    plots = sorted((Path(base) / "plots").glob("*.png"))
    if plots:
        out += ["", "Plots (seed 0): " + ", ".join(f"[{p.stem}](plots/{p.name})" for p in plots) + "."]
    return "\n".join(out) + "\n"


def _snap_table(rows: list[dict]) -> list[str]:
    out = ["| variant | learned | formula found | with 0 terms | with 1–4 terms | route pick is the family | won on size "
           "| other family passed | PySR found | near formula within 3% (regression, of the family) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        ls = [a for a in rows if a["variant"] == v and a["learned"]]
        if not [a for a in rows if a["variant"] == v]:
            continue
        f = [a for a in ls if a["formula_found"]]
        out.append(f"| {v} | {len(ls)} | {_cnt(len(f), len(ls))} | {sum(a['found_terms'] == 0 for a in f)} | "
                   f"{sum((a['found_terms'] or 0) > 0 for a in f)} | {sum(a['route_pick'] == a['family'] for a in ls)} | "
                   f"{sum(bool(a['won_on_size']) for a in ls)} | {sum(bool(a['other_families']) for a in ls)} | "
                   f"{sum(bool(a.get('pysr_found')) for a in ls)} | "
                   f"{sum(_near_ok(a) and not a['formula_found'] for a in ls if _is_regression(a))} |")
    return out


def _predictions_md(dev: list[dict], held: list[dict]) -> str:
    rows = []

    def add(text, measured, met):
        rows.append(f"| {text} | {measured} | {'–' if met is None else 'met' if met else 'not met'} |")

    seeds = sorted({a["seed"] for a in dev})
    small = [sum(not a["learned"] for a in dev if a["variant"] == "relu" and a["seed"] == s) for s in seeds]
    add("on each development seed, at least 3 of the 9 small ReLU nets do not learn",
        ", ".join(str(k) for k in small) or "–", bool(small) and all(k >= 3 for k in small))
    for v in ("relu_wide", "sigmoid", "tanh"):
        per = [sum(a["learned"] for a in dev if a["variant"] == v and a["seed"] == s) for s in seeds]
        if any(a["variant"] == v for a in dev):
            add(f"{v} learns at least 8 of 9 tasks on at least 2 of the 3 seeds", ", ".join(map(str, per)),
                sum(k >= 8 for k in per) >= 2)
    ls = [a for a in dev if a["learned"]]
    f = [a for a in ls if a["formula_found"]]
    add("formula found on at least 70% of the learned development nets", _pct(len(f), len(ls)),
        bool(ls) and len(f) >= 0.7 * len(ls))
    ratio = [a for a in f if a.get("ast4_nodes") and a.get("ast5_nodes")]
    small5 = sum(a["ast4_nodes"] >= 5 * a["ast5_nodes"] for a in ratio)
    add("where found, ast5's program is at least 5× smaller than ast4's", _pct(small5, len(ratio)),
        bool(ratio) and small5 == len(ratio))
    bump = sum(a["formula_found"] for a in ls if a["task"] == "bump")
    add("bump: formula found on at least one net", str(bump), bump >= 1)
    sig = sum(a["formula_found"] for a in ls if a["variant"] == "sigmoid")
    tan = sum(a["formula_found"] for a in ls if a["variant"] == "tanh")
    add("tanh's formula-found count is within 3 of sigmoid's", f"{tan} vs {sig}", abs(tan - sig) <= 3)
    py = sum(bool(a.get("pysr_found")) for a in dev)
    add("PySR finds the formula on fewer development nets than ast5, and on at least 5", f"{py} vs {len(f)}",
        py < len(f) and py >= 5)
    if held:
        hl = [a for a in held if a["learned"]]
        hf = [a for a in hl if a["formula_found"]]
        add("held-out: formula found on at least 60% of the learned held-out nets", _pct(len(hf), len(hl)),
            bool(hl) and len(hf) >= 0.6 * len(hl))
        for t in ("ellipse", "decay"):
            k = sum(a["task"] == t for a in hf)
            add(f"held-out: formula found on at least one {t} net", str(k), k >= 1)
    return "\n".join(["## Predictions", "",
                      "Written in the design before any tool code. Counts are over learned nets.", "",
                      "| prediction | measured | met |", "|---|---|---|", *rows]) + "\n"


def _before_after(dev: list[dict], base: Path) -> tuple[str, bool | None]:
    p = Path(base) / "before_minimax.json"
    if not p.exists():
        return "– (no saved list)", None
    before = json.loads(p.read_text(encoding="utf-8"))["runs"]
    now = {f"{a['variant']}/{a['task']}/{a['seed']}": a for a in dev}
    both = [(b, now[k]) for k, b in before.items() if k in now]
    fb = sum(bool(b["formula_found"]) for b, _ in both)
    fn = sum(bool(a["formula_found"]) for _, a in both)
    passing = [(b, a) for b, a in both if b.get("ast5_pass") and a.get("ast5_pass")
               and b.get("ast5_nodes") is not None and a.get("ast5_nodes") is not None]
    larger = sum(a["ast5_nodes"] > b["ast5_nodes"] for b, a in passing)
    return f"found {fb} before, {fn} now; larger on {larger} of {len(passing)}", fn >= fb and larger == 0


def _gate_predictions_md(dev: list[dict], base: Path) -> str:
    rows = []

    def add(text, measured, met):
        rows.append(f"| {text} | {measured} | {'–' if met is None else 'met' if met else 'not met'} |")

    dist = [a for a in dev if a["variant"] == "sigmoid" and a["task"] == "dist"]
    k = sum(bool(a["formula_found"]) for a in dist)
    add("with minimax, the formula is found on at least 2 of the 3 sigmoid dist nets", _pct(k, len(dist)),
        bool(dist) and k >= 2)
    reg = [a for a in dev if a["learned"] and _is_regression(a)]
    ok = sum(_near_ok(a) for a in reg)
    add("on at least 90% of the learned regression development nets, the formula is found or the near formula is of "
        "the task's family and within 3%", _pct(ok, len(reg)), bool(reg) and ok >= 0.9 * len(reg))
    measured, met = _before_after(dev, base)
    add("on the nets run before minimax, ast5 finds the formula on at least as many, and is never larger where both "
        "pass", measured, met)
    return "\n".join(["## Predictions (near formulas and gate view)", "",
                      "Written in `docs/design/2026-10-31-formula-residual-gate-view-design.md` before any of its code.",
                      "", "| prediction | measured | met |", "|---|---|---|", *rows]) + "\n"


def build_irrational_report(base: Path) -> str:
    rows = load_rows(base)
    dev = [a for a in rows if not a["heldout"]]
    held = [a for a in rows if a["heldout"]]
    parts = ["# Irrational functions", "",
             "Nets trained on exp, log, sqrt, x², 1/x, a circle, a distance, a bump and a product, as ReLU, wide ReLU "
             "(4× units, 3× steps), sigmoid and tanh nets. `ast5` is `ast4` plus the family and conic routes. "
             "Design: `docs/design/2026-10-30-irrational-functions-design.md`. Generated by `nndc irr-report` from "
             "each run's `irr_analysis.json`.", "",
             _results_md(rows), _model_md(rows, base),
             "\n".join(["## Snapping to formulas", "", "Development nets.", "", *_snap_table(dev)]) + "\n",
             _predictions_md(dev, held), _gate_predictions_md(dev, base)]
    if held:
        freezes = sorted({a["freeze"] for a in held if a["freeze"]})
        parts.append("\n".join(["## Held-out", "",
                                f"Built after the freeze and run once (ast5, freeze {', '.join(freezes) or '–'}).", "",
                                *_snap_table(held)]) + "\n")
    return "\n".join(parts)
