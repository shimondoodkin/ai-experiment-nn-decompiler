"""Step-4 comparison section of REPORT.md: methods side by side, Iris, the checker loop, fix and reinsert.

Everything here is computed from the run folders (results.json, *_info.json, *_meta.json, *_checks.log,
repair/results.json); nothing is hand-written.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .run import RUNS, all_runs, is_step1

METHODS = [("Claude (dump)", "agent"), ("Claude (input/output only)", "io_agent"), ("AST tool", "ast"),
           ("PySR", "pysr"), ("PySR+snap", "pysr_snap"), ("tree", "tree"), ("regions", "regions")]
FIX_ARMS = [("Claude fix (with program)", "fix"), ("Claude fix (cases only)", "fixio"), ("tree on cases", "r_tree"),
            ("fine-tune", "w_ft"), ("fine-tune + anchors", "w_ft_anchor"), ("last-layer LP", "w_lp"),
            ("retrain (clean labels)", "w_retrain")]
EXTRA_SEEDS = (2, 3, 4, 5)


def _results(rd: Path) -> dict | None:
    p = rd / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pick(versions: dict, key: str) -> tuple[str | None, bool]:
    """(version name, audit-flagged) for a method: best valid attempt for agents, the file otherwise."""
    from .report import _best_attempt

    if key == "agent":
        best = _best_attempt(versions, valid_only=True)
        return best, best is None and _best_attempt(versions) is not None
    if key == "io_agent":
        names = [k for k in versions if k.startswith("io_agent_")]
        valid = [k for k in names if not (versions[k].get("meta") or {}).get("forbidden_paths")]
        if not valid:
            return None, bool(names)
        return min(valid, key=lambda k: (not versions[k]["check"]["snap_pass"],
                                         (versions[k]["size"] or {}).get("nodes", 10**9))), False
    return (key if key in versions else None), False


def _seconds(rd: Path, versions: dict, key: str, name: str) -> float | None:
    if key in ("agent", "io_agent"):
        return (versions[name].get("meta") or {}).get("duration_s")
    info = rd / f"{'pysr' if key == 'pysr_snap' else key}_info.json"
    return json.loads(info.read_text(encoding="utf-8")).get("seconds") if info.exists() else None


def _med(xs: list, fmt: str = ".0f") -> str:
    xs = [x for x in xs if x is not None]
    return format(statistics.median(xs), fmt) if xs else "–"


def _step1_runs(root: Path, split: str) -> list[Path]:
    runs = [rd for rd in all_runs(root) if is_step1(rd) and _results(rd)]
    if split == "heldout":
        return [rd for rd in runs if rd.name == "seed2"]
    if split == "insample":
        return [rd for rd in runs if rd.name != "seed2"]
    return runs


def method_table(root: Path, split: str) -> str:
    runs = _step1_runs(root, split)
    n = len(runs)
    rows = ["| method | passes check | exact | recovered | median nodes | median seconds | no program |",
            "|---|---|---|---|---|---|---|"]
    notes = []
    for label, key in METHODS:
        passes = exact = rec = missing = flagged = 0
        nodes, secs = [], []
        for rd in runs:
            vs = _results(rd)["versions"]
            name, flag = _pick(vs, key)
            flagged += flag
            if name is None:
                missing += 1
                continue
            v = vs[name]
            passes += bool(v["check"]["snap_pass"])
            exact += bool(v["check"]["exact_pass"])
            rec += bool(v["task"]["recovered"])
            nodes.append((v["size"] or {}).get("nodes"))
            secs.append(_seconds(rd, vs, key, name))
        if key == "io_agent" and flagged * 3 > n:
            rows.append(f"| {label} | invalid (audit) | | | | | |")
            continue
        rows.append(f"| {label} | {passes}/{n} | {exact}/{n} | {rec}/{n} | {_med(nodes)} | {_med(secs)} | {missing} |")
        if flagged:
            notes.append(f"{label}: {flagged} run(s) excluded by the audit (counted under no program).")
    return "\n".join(rows) + ("\n\n" + "\n".join(f"- {s}" for s in notes) if notes else "") + "\n"


def recovery_matrix(root: Path) -> str:
    runs = _step1_runs(root, "all")
    rows = ["| run | " + " | ".join(lab for lab, _ in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for rd in runs:
        res = _results(rd)
        cells = []
        for _, key in METHODS:
            name, _ = _pick(res["versions"], key)
            cells.append("–" if name is None else "✓" if res["versions"][name]["task"]["recovered"] else "✗")
        rows.append(f"| {res['run']} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def what_this_shows(root: Path) -> list[str]:
    runs = _step1_runs(root, "all")

    def count(key: str, field: str) -> tuple[int, int]:
        hit = have = 0
        for rd in runs:
            vs = _results(rd)["versions"]
            name, _ = _pick(vs, key)
            if name:
                have += 1
                hit += bool(vs[name]["task"]["recovered"] if field == "recovered" else vs[name]["check"][field])
        return hit, have

    def nodes(key: str) -> list:
        out = []
        for rd in runs:
            vs = _results(rd)["versions"]
            name, _ = _pick(vs, key)
            if name:
                out.append((vs[name]["size"] or {}).get("nodes"))
        return out

    n = len(runs)
    (cd, _), (io, io_have) = count("agent", "recovered"), count("io_agent", "recovered")
    (p1, p1h), (p2, p2h) = count("pysr", "recovered"), count("pysr_snap", "recovered")
    regions_missing = n - count("regions", "snap_pass")[1]
    return [f"**Reading the weights:** Claude recovers the formula on {cd}/{n} nets from the dump, and on "
            f"{io}/{n} from input/output samples only ({io_have} with an io attempt).",
            f"**Snapping in search:** PySR recovers {p1}/{p1h} raw and {p2}/{p2h} after this repo's snapping.",
            f"**Exact vs short:** exact region programs have a median of {_med(nodes('regions'))} nodes "
            f"(no program within budget on {regions_missing} nets), against {_med(nodes('agent'))} for Claude and "
            f"{_med(nodes('ast'))} for the AST tool."]


def _flower_accuracy(rd: Path, spec, name: str) -> float | None:
    from experiments.datasets import table

    from .check import program_decision
    from .lang import evaluate, parse_program, read_program

    X, y = table(spec.task)
    raw = evaluate(parse_program(read_program(rd / f"{name}.py"), spec.arg_names), X)
    return float((program_decision(spec, raw) == y).mean())


def iris_table(root: Path) -> str:
    from .check import check_program
    from .lang import read_program
    from .run import load_run

    rows = ["| run | method | verdict | nodes | confident disagreements, checker seeds 2/3/4/5 | flower accuracy |",
            "|---|---|---|---|---|---|"]
    for rd in all_runs(root):
        if is_step1(rd) or not _results(rd):
            continue
        spec, mlp = load_run(rd)
        res = _results(rd)
        for label, key in METHODS:
            name, flag = _pick(res["versions"], key)
            if name is None:
                rows.append(f"| {res['run']} | {label} | {'audit-flagged' if flag else 'no program'} | | | |")
                continue
            v = res["versions"][name]
            c = v["check"]
            verdict = "ERROR" if c["error"] else "ex" if c["exact_pass"] else "sn" if c["snap_pass"] else "FAIL"
            src = read_program(rd / f"{name}.py")
            extra = "/".join(str(check_program(src, mlp, spec, seed=s).disagreements_outside) for s in EXTRA_SEEDS) \
                if not c["error"] else "–"
            acc = _flower_accuracy(rd, spec, name) if not c["error"] else None
            rows.append(f"| {res['run']} | {label} | {verdict} | {(v['size'] or {}).get('nodes', '–')} | {extra} | "
                        f"{'–' if acc is None else f'{100 * acc:.1f}%'} |")
    return "\n".join(rows) + "\n"


def _log_lines(path: Path) -> list[str]:
    return [ln.split("\t")[1] for ln in path.read_text(encoding="utf-8").splitlines() if ln.count("\t") >= 2]


def loop_counts(roots: list[Path]) -> str:
    rows = ["| method | nets | runs | median checks | max checks | runs with a FAIL | started with FAIL | "
            "max FAILs before the first PASS |", "|---|---|---|---|---|---|---|---|"]
    for label, prefix in (("Claude (dump)", "agent"), ("Claude (input/output only)", "io_agent")):
        for group in ("formula", "Iris"):
            stats = []
            for root in roots:
                for rd in all_runs(root):
                    if is_step1(rd) != (group == "formula"):
                        continue
                    for log in sorted(rd.glob(f"{prefix}_*_checks.log")):
                        verdicts = _log_lines(log)
                        if not verdicts:
                            continue
                        first_pass = next((i for i, v in enumerate(verdicts) if v.startswith("PASS")), len(verdicts))
                        stats.append((len(verdicts), any(v.startswith("FAIL") for v in verdicts),
                                      verdicts[0].startswith("FAIL"),
                                      sum(v.startswith("FAIL") for v in verdicts[:first_pass])))
            if stats:
                rows.append(f"| {label} | {group} | {len(stats)} | {_med([s[0] for s in stats])} | "
                            f"{max(s[0] for s in stats)} | {sum(s[1] for s in stats)} | {sum(s[2] for s in stats)} | "
                            f"{max(s[3] for s in stats)} |")
    return "\n".join(rows) + "\n"


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _arm_rows(entries: list[dict]) -> list[str]:
    rows = []
    for label, key in FIX_ARMS:
        found = [e for e in entries if e["arm"] == key or e["arm"].startswith(f"{key}_")
                 and key in ("fix", "fixio") and e["arm"].split("_")[-1].isdigit()]
        found = [e for e in found if not (key == "fix" and e["arm"].startswith("fixio"))]
        if not found:
            continue
        ok = [e["row"] for e in found if not e["row"].get("error")]

        def med(path: tuple[str, str]):
            xs = [r.get(path[0], {}).get(path[1]) for r in ok]
            xs = [x for x in xs if x is not None]
            return statistics.median(xs) if xs else None

        flagged = sum(sum(t["flagged"] for t in (r.get("added_tests") or [])) for r in ok)
        rein = [r["reinsertion"] for r in ok if r.get("reinsertion")]
        rein_s = f"{sum(bool(x.get('exact_pass') or x.get('snap_pass')) for x in rein)}/{len(rein)}" if rein else "–"
        errors = len(found) - len(ok)
        rows.append(f"| {label} | {len(found)} | {_fmt(med(('repair', 'efficacy')))} | "
                    f"{_fmt(med(('heldout', 'generalization')))} | {_fmt(med(('heldout', 'drawdown')))} | "
                    f"{_fmt(med(('heldout', 'accuracy')))} | {_fmt(med(('oor', 'accuracy')))} | "
                    f"{_med([r.get('nodes') for r in ok])} | {flagged} | {rein_s} |"
                    + (f" ({errors} infeasible/error)" if errors else ""))
    return rows


def fix_table(roots: list[Path]) -> str:
    groups: dict[str, list[dict]] = {"planted bug (P1)": [], "natural bug (P2)": [], "Iris (P3)": []}
    for root in roots:
        for rd in all_runs(root):
            p = rd / "repair" / "results.json"
            if not p.exists():
                continue
            g = "planted bug (P1)" if (rd / "defect.json").exists() else "natural bug (P2)" if is_step1(rd) \
                else "Iris (P3)"
            for arm, row in json.loads(p.read_text(encoding="utf-8")).items():
                groups[g].append({"run": rd, "arm": arm, "row": row})
    out = []
    head = ("| arm | nets | efficacy (repair set) | generalization (held-out failures fixed) | drawdown "
            "(held-out rights broken) | held-out accuracy | out-of-range accuracy | median nodes | "
            "flagged added tests | reinsertion PASS |\n|---|---|---|---|---|---|---|---|---|---|")
    for g, entries in groups.items():
        if entries:
            out.append(f"**{g}**, medians over nets\n\n{head}\n" + "\n".join(_arm_rows(entries)) + "\n")
    return "\n".join(out) if out else "(no fix-experiment results yet)\n"


def _fix_diffs(roots: list[Path]) -> str:
    from .repair.run import faithful_program
    from .run import agent_attempts

    parts = []
    for root in roots:
        for rd in all_runs(root):
            fixes = agent_attempts(rd, "fix")
            if not fixes or not is_step1(rd):
                continue
            try:
                faithful = faithful_program(rd).read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            fix = fixes[-1][1].read_text(encoding="utf-8")
            parts.append(f"#### {rd.parent.name}/{rd.name} ({rd.parent.parent.name})\n\nDecompiled (faithful to the "
                         f"net):\n\n```python\n{faithful}```\n\nFixed by Claude:\n\n```python\n{fix}```\n")
    return "\n".join(parts)


def comparison_md(formula_root: Path = RUNS, iris_root: Path = RUNS, bug_root: Path = Path("runs_bugs")) -> str:
    roots = [r for r in dict.fromkeys([Path(formula_root), Path(iris_root), Path(bug_root)]) if r.exists()]
    parts = ["## Comparison (step 4)\n",
             "Each method's program for each formula net, checked against the net and graded against the true "
             "task. Claude (dump) is the best valid step-2 attempt; Claude (input/output only) saw 2,000 samples "
             "and a checker without the net's boundary points; PySR, tree and regions are the baselines in "
             "`docs/design/2026-10-27-step4-comparison-design.md`. Audit-flagged agent runs are excluded.\n",
             "### Held-out nets (seed 2)\n", method_table(formula_root, "heldout"),
             "### In-sample nets\n", method_table(formula_root, "insample"),
             "### Which method recovered which formula\n", recovery_matrix(formula_root),
             "\n".join(f"- {s}" for s in what_this_shows(formula_root)) + "\n"]
    iris = iris_table(iris_root)
    if iris.count("\n") > 2:
        parts += ["### Iris nets\n", iris]
    parts += ["### The checker loop\n", "Calls to `check.py` per agent run (`*_checks.log`).\n", loop_counts(roots)]
    parts += ["### Fix and reinsert (experiment R)\n",
              "The net is wrong on the task; each arm fixes it from the repair set and is judged on held-out "
              "cases. Program arms are compiled back into a net (reinsertion) and that net is checked against "
              "the program.\n", fix_table(roots)]
    diffs = _fix_diffs(roots)
    if diffs:
        parts += ["### Readable diffs of network repairs\n", diffs]
    return "\n".join(parts) + "\n"
