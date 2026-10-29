"""REPORT.md from all runs, and the spike gate."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .compare import comparison_md
from .run import RUNS, all_runs, is_step1, read_git
from .spec import Spec

SPIKE_RUNS = ["xor/seed0", "absdiff/seed0", "neareq/seed0"]
BUG_ROOT = Path("runs_bugs")   # planted-bug nets of the fix experiment (step 4)


def with_real_names(src: str, spec: Spec) -> str:
    names = {i.name: i.real_name for i in spec.inputs}
    return re.sub(r"\b(x\d+)\b", lambda m: names.get(m.group(1), m.group(1)), src)


def _verdict(c: dict) -> str:
    if c.get("error"):
        return "ERROR"
    return "PASS exact" if c["exact_pass"] else "PASS snap" if c["snap_pass"] else "FAIL"


def _num(v, fmt: str = ".3g") -> str:
    return "" if v is None else v if isinstance(v, str) else format(v, fmt)


def _training(res: dict) -> str:
    t = res.get("train") or {}
    if t.get("kind") == "hand":
        return "hand"
    q = t.get("quality", {})
    main = f"acc {q['accuracy']:.4f}" if "accuracy" in q else f"R² {q.get('r2', float('nan')):.4f}"
    return main + ("" if t.get("learned") else " (NOT learned)")


def _audit(meta: dict | None) -> tuple[str, str]:
    if meta is None:
        return "", ""
    if meta.get("outside_paths"):
        audit = f"OUTSIDE PATHS ({len(meta['outside_paths'])})"
    elif not meta.get("transcript_found"):
        audit = "no transcript"
    else:
        audit = "ok"
    tok = meta.get("tokens") or {}
    cost = []
    if meta.get("duration_s") is not None:
        cost.append(f"{meta['duration_s']:.0f}s")
    if tok:
        cost.append(f"{tok.get('input', 0)}+{tok.get('output', 0)} tok")
    return audit, " ".join(cost)


def _best_attempt(versions: dict, valid_only: bool = False) -> str | None:
    agents = [k for k in versions if k.startswith("agent_")
              and not (valid_only and ((versions[k].get("meta") or {}).get("outside_paths")))]
    if not agents:
        return None
    return min(agents, key=lambda k: (not versions[k]["check"]["snap_pass"],
                                      (versions[k]["size"] or {}).get("nodes", 10**9)))


def build_report(root: Path = RUNS, compare_to: Path | None = None) -> str:
    """With compare_to (a root of ReLU runs), the step-4 comparison is replaced by the formula-vs-net and the
    activation comparison sections, for a root of non-ReLU runs."""
    rows = ["| run | training | version | check | label | max err | disagree outside / inside (zone) | "
            "task in / out | recovered | nodes (% of dump) | audit | Claude |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    sections = []
    for rd in all_runs(root):
        res_path = rd / "results.json"
        if not res_path.exists():
            continue
        res = json.loads(res_path.read_text(encoding="utf-8"))
        spec = Spec.load(rd / "spec.json")
        vs = res["versions"]
        dump_nodes = (vs["dump"]["size"] or {}).get("nodes")
        for name, v in vs.items():
            c, t = v["check"], v["task"]
            nodes = (v["size"] or {}).get("nodes")
            pct = f" ({100 * nodes / dump_nodes:.0f}%)" if nodes and dump_nodes else ""
            dis = "" if spec.output == "regression" else \
                f"{c['disagreements_outside']} / {c['disagreements_inside']} ({100 * c['zone_fraction']:.2f}%)"
            audit, cost = _audit(v.get("meta"))
            rows.append(f"| {res['run']} | {_training(res)} | {name} | {_verdict(c)} | {c['label']} | "
                        f"{_num(c.get('max_abs_err'))} | {dis} | {_num(t.get('in_range'))} / "
                        f"{_num(t.get('out_of_range'))} | "
                        f"{('yes' if t.get('recovered') else 'no') if is_step1(rd) else 'n/a'} | "
                        f"{_num(nodes, 'd')}{pct} | {audit} | {cost} |")
        best = _best_attempt(vs)
        if best:
            dump_src = (rd / "dump.py").read_text(encoding="utf-8")
            best_src = (rd / f"{best}.py").read_text(encoding="utf-8", errors="replace")
            notes_path = rd / f"{best}_notes.md"
            notes = notes_path.read_text(encoding="utf-8", errors="replace").strip() if notes_path.exists() \
                else "(no notes)"
            # nest the agent's own headings under this run's section
            notes = re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", notes, flags=re.M)
            ast_md = ""
            if (rd / "ast.py").exists():
                ast_src = (rd / "ast.py").read_text(encoding="utf-8", errors="replace")
                ast_md = f"AST tool (no LLM):\n\n```python\n{with_real_names(ast_src, spec)}```\n\n"
            sections.append(f"## {res['run']}: dump vs {best}\n\n```python\n{with_real_names(dump_src, spec)}```\n\n"
                            f"```python\n{with_real_names(best_src, spec)}```\n\n{ast_md}<details><summary>notes"
                            f"</summary>\n\n{notes}\n\n</details>\n")
    head = ("# Report\n\nGenerated by `nndc report`. Checks compare each program with the original net "
            "(never the target); `task in / out` grades against the true task inside and outside the "
            "training range (accuracy, or max abs error for regression). Input names are shown with "
            "their real names here; the refactorer only saw `x0, x1, …`.\n\n")
    beats = _beats_the_net(root)
    beats_md = "\n".join(beats) if beats else "(none yet)"
    return (head + LEGEND + "\n## Where the refactored program beats the network\n\n" + BEATS_INTRO + beats_md
            + "\n\n" + reproduction_md(Path(root))
            + (comparison_md(Path(root), Path(root), BUG_ROOT) if compare_to is None
               else flagged_reproduction_md(Path(root)) + formula_md(Path(root))
               + activation_compare_md(Path(compare_to), Path(root)) + kink_md(Path(root)))
            + _datasets_md(Path(root)) + "## Results\n\n" + "\n".join(rows) + "\n\n" + "\n".join(sections))


def _results(rd: Path) -> dict | None:
    p = rd / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _cell(rd: Path, v: dict | None) -> str:
    """Verdict, recovered (formula tasks only) and nodes of one version."""
    if v is None:
        return "–"
    rec = ("rec" if v["task"].get("recovered") else "not rec") if is_step1(rd) else "n/a"
    return f"{_verdict(v['check'])}, {rec}, {_num((v['size'] or {}).get('nodes'), 'd')} nodes"


def formula_md(root: Path) -> str:
    """The intended formula (formula.py) judged against each net, next to the agent's and the tool's programs."""
    from experiments.tasks import get_task

    rows = ["| run | formula vs net | max err (% of output range) / disagreements outside, inside | "
            "formula task in / out | best agent | AST tool |", "|---|---|---|---|---|---|"]
    cases = []
    for rd in all_runs(root):
        res = _results(rd)
        if res is None or "formula" not in res["versions"]:
            continue
        spec = Spec.load(rd / "spec.json")
        vs = res["versions"]
        f, c = vs["formula"], vs["formula"]["check"]
        if spec.output == "regression" and c.get("margin"):
            size = f"{_num(c.get('max_abs_err'))} ({100 * spec.snap_tol_frac * c['max_abs_err'] / c['margin']:.1f}%)"
        else:
            size = f"{c.get('disagreements_outside')}, {c.get('disagreements_inside')}"
        best = _best_attempt(vs)
        rows.append(f"| {res['run']} | {_verdict(c)} | {size} | {_num(f['task'].get('in_range'))} / "
                    f"{_num(f['task'].get('out_of_range'))} | {_cell(rd, vs.get(best) if best else None)} | "
                    f"{_cell(rd, vs.get('ast'))} |")
        cex = c.get("counterexamples") or []
        if cex:
            target = get_task(spec.task).target
            lines = ["| inputs | net | formula | true |", "|---|---|---|---|"]
            for e in cex[:5]:
                x = np.array([[e["inputs"][n] for n in spec.arg_names]], dtype=np.float64)
                truth = float(np.asarray(target(x), dtype=np.float64)[0])
                ins = ", ".join(f"{i.real_name}={e['inputs'][i.name]:.4g}" for i in spec.inputs)
                lines.append(f"| {ins} | {_num(e['net'])} | {_num(e['program'])} | {_num(truth)} |")
            cases.append(f"**{res['run']}**: where the formula and the net disagree most\n\n" + "\n".join(lines) + "\n")
    if len(rows) == 2:
        return ""
    head = ("## Formula vs net\n\nThe intended formula of each task (`formula.py`), checked against the net like any "
            "other version but never used to harden the other checks. A FAIL means the net rounds or shifts the "
            "formula beyond the checker's tolerance; the cases below show where.\n\n")
    return head + "\n".join(rows) + "\n\n" + "\n".join(cases) + "\n"


def activation_compare_md(relu_root: Path, root: Path) -> str:
    """One row per task and seed present in both roots: the net, the best agent program and the AST program."""
    rows = ["| run | learned (ReLU / this) | best agent, ReLU | best agent, this | AST, ReLU | AST, this |",
            "|---|---|---|---|---|---|"]
    for rd in all_runs(root):
        other = Path(relu_root) / rd.parent.name / rd.name
        a, b = _results(other), _results(rd)
        if a is None or b is None:
            continue
        cells = []
        for res, d in ((a, other), (b, rd)):
            best = _best_attempt(res["versions"])
            cells.append((_cell(d, res["versions"].get(best) if best else None), _cell(d, res["versions"].get("ast"))))
        learned = " / ".join("yes" if (r.get("train") or {}).get("learned") else "no" for r in (a, b))
        rows.append(f"| {b['run']} | {learned} | {cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")
    if len(rows) == 2:
        return ""
    return (f"## Compared with the ReLU runs\n\nThe same task, seed and training settings, with ReLU "
            f"(`{Path(relu_root).as_posix()}`) and with this root's activation.\n\n" + "\n".join(rows) + "\n\n")


def _kink_heldout(rd: Path) -> bool:
    """The kink-merge held-out set: seeds 3-5, built after its freeze."""
    return rd.name in ("seed3", "seed4", "seed5")


def _kink_line(line: dict, spec: Spec) -> str:
    from .symnet import Affine

    z = Affine({k: float(f"{c:.3g}") for k, c in line["normal"].items()}, -float(f"{line['t']:.3g}"))
    return f"{_num(line['strength'])} · max({with_real_names(z.to_src(), spec)}, 0)"


def kink_md(root: Path) -> str:
    """The kink-merge tool (ast2): per run, the kink counts and the kept lines; then ast2 next to formula.py."""
    rows = ["| run | split | route | ast2 vs net | recovered | nodes | kinks → lines → nearly cancelled → kept "
            "| kept lines |", "|---|---|---|---|---|---|---|---|"]
    progs = []
    for rd in all_runs(root):
        res = _results(rd)
        if res is None or "ast2" not in res["versions"]:
            continue
        spec = Spec.load(rd / "spec.json")
        v = res["versions"]["ast2"]
        tp = rd / "ast2_trace.json"
        tr = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
        km = tr.get("kink_merge") or {}
        w, per = km.get("winner"), km.get("per_offset") or []
        if w:
            po = next((p for p in per if p["off"] == w["off"]), None)
            counts = f"{po['kinks']} → {po['lines']} → {po['cancelled']} → {w['m']}" if po else ""
            kept = "; ".join(_kink_line(ln, spec) for ln in w["lines"])
        else:
            counts = (f"{per[0]['kinks']} → {per[0]['lines']} → {per[0]['cancelled']} → –" if per
                      else km.get("applies") or "–")
            kept = ""
        rec = _yn(bool(v["task"].get("recovered"))) if is_step1(rd) else "n/a"
        rows.append(f"| {res['run']} | {'held-out' if _kink_heldout(rd) else 'development'} | {tr.get('winner', '')} | "
                    f"{_verdict(v['check'])} | {rec} | {_num((v['size'] or {}).get('nodes'), 'd')} | {counts} | {kept} |")
        if (rd / "formula.py").exists() and (rd / "ast2.py").exists():
            a2 = with_real_names((rd / "ast2.py").read_text(encoding="utf-8"), spec).strip()
            fo = with_real_names((rd / "formula.py").read_text(encoding="utf-8"), spec).strip()
            progs.append(f"**{res['run']}**: `ast2`, then `formula.py`\n\n```python\n{a2}\n```\n\n```python\n{fo}\n```\n")
    if len(rows) == 2:
        return ""
    out = ["## Kink merge (ast2)\n",
           "The kink-merge tool turns each sigmoid into straight pieces, merges the kinks that lie on one line, keeps the "
           "strongest lines and refits them to the net (`ast2`). Development runs are seeds 0–2; held-out runs "
           "(seeds 3–5) were built after the tool was frozen. *kinks → lines → nearly cancelled → kept* are the "
           "counts for the winning merge tolerance.\n"]
    rep = reproduction(root, name="ast2", heldout=_kink_heldout)
    for split, label in (("heldout", "held-out"), ("distill", "development")):
        rs = [r for r in rep if r["split"] == split]
        if rs:
            ok = sum(r["ast_check"].startswith("PASS") for r in rs)
            routes = sum(r["route"] == "kink-merge" for r in rs)
            vs = [r for r in rs if r["reproduces"] is not None]
            out.append(f"- **{label}**: {ok} of {len(rs)} pass, kink-merge won {routes}, recovered the formula in "
                       f"{sum(r['ast_recovered'] for r in rs)}; reproduces Claude's best valid attempt in "
                       f"{sum(r['reproduces'] for r in vs)} of {len(vs)}")
    return "\n".join(out) + "\n\n" + "\n".join(rows) + "\n\n" + "\n".join(progs) + "\n"


def _datasets_md(root: Path) -> str:
    """Pointer to dataset runs (no known formula), which have their own hand-written report."""
    reports = sorted(Path(root).glob("*/REPORT.md"))
    if not reports:
        return ""
    links = ", ".join(f"[{p.as_posix()}]({p.as_posix()})" for p in reports)
    return ("## Real datasets\n\nDataset runs (Iris) have no known formula: their task columns below compare with "
            "the label of the nearest table row, and `recovered` is n/a. They are left out of the step-3 "
            f"reproduction and the beats-the-network list above. Their results and held-out runs are in {links}.\n\n")


def reproduction(root: Path = RUNS, valid_only: bool = True, name: str = "ast", heldout=None) -> list[dict]:
    """Per run with a tool version (`name`): does the deterministic tool reproduce Claude's best valid attempt?
    valid_only=False also counts attempts with outside-path audit flags. heldout(rd) marks held-out runs
    (default: seed 2, the step-3 protocol)."""
    rows = []
    for rd in all_runs(root):
        res_path = rd / "results.json"
        if not is_step1(rd) or not res_path.exists():
            continue
        res = json.loads(res_path.read_text(encoding="utf-8"))
        vs = res["versions"]
        if name not in vs:
            continue
        a = vs[name]
        trace_path = rd / f"{name}_trace.json"
        route = json.loads(trace_path.read_text(encoding="utf-8"))["winner"] if trace_path.exists() else None
        held = rd.name == "seed2" if heldout is None else heldout(rd)
        row = {"run": res["run"], "split": "heldout" if held else "distill", "route": route,
               "ast_git": read_git(rd).get(name),
               "ast_check": _verdict(a["check"]), "ast_recovered": bool(a["task"]["recovered"]),
               "ast_nodes": (a["size"] or {}).get("nodes")}
        best = _best_attempt(vs, valid_only=valid_only)
        if best is None:
            row.update({"agent_check": None, "agent_recovered": None, "agent_nodes": None, "reproduces": None})
        else:
            g = vs[best]
            an, gn = row["ast_nodes"], (g["size"] or {}).get("nodes")
            row.update({"agent_check": _verdict(g["check"]), "agent_recovered": bool(g["task"]["recovered"]),
                        "agent_nodes": gn})
            row["reproduces"] = bool(a["check"]["snap_pass"] and row["ast_recovered"] == row["agent_recovered"]
                                     and an is not None and gn is not None and an <= max(1.5 * gn, gn + 5))
        rows.append(row)
    return rows


def _yn(b) -> str:
    return "" if b is None else "yes" if b else "no"


def reproduction_md(root: Path) -> str:
    rows = reproduction(root)
    if not rows:
        return ""
    out = ["## Reproduction (step 3)\n"]
    held = sorted({r["ast_git"] for r in rows if r["split"] == "heldout" and r["ast_git"]})
    if held:
        out.append(f"Tool version on the held-out runs (from each run's `run_git.txt`): {', '.join(held)}.\n")
    out.append("Does the deterministic AST tool (no LLM) reproduce Claude's best valid refactoring? Reproduces = "
               "passes the check, same recovered status as Claude, and at most max(1.5x, +5) of Claude's nodes. "
               "Held-out runs (seed 2) were never studied while building the tool; in-sample runs were.\n")
    for split, label in (("heldout", "held-out (headline)"), ("distill", "in-sample")):
        rs = [r for r in rows if r["split"] == split and r["reproduces"] is not None]
        if rs:
            out.append(f"- **{label}**: {sum(r['reproduces'] for r in rs)} of {len(rs)} reproduced; "
                       f"tool recovered the true formula in {sum(r['ast_recovered'] for r in rs)}, "
                       f"Claude in {sum(r['agent_recovered'] for r in rs)}")
    wins: dict[str, int] = {}
    for r in rows:
        wins[r["route"] or "?"] = wins.get(r["route"] or "?", 0) + 1
    out.append(f"- **winning routes**: {', '.join(f'{k} {v}' for k, v in sorted(wins.items()))}")
    only = [r["run"] for r in rows if r["agent_recovered"] and not r["ast_recovered"]]
    out.append(f"- **only Claude recovered the formula**: {', '.join(only) or 'none'}\n")
    out.append("| run | split | tool check | tool recovered | tool nodes | Claude check | Claude recovered | "
               "Claude nodes | reproduces | route |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        rep = "no valid Claude attempt" if r["reproduces"] is None else _yn(r["reproduces"])
        out.append(f"| {r['run']} | {r['split']} | {r['ast_check']} | {_yn(r['ast_recovered'])} | {r['ast_nodes']} | "
                   f"{r['agent_check'] or ''} | {_yn(r['agent_recovered'])} | {r['agent_nodes'] or ''} | {rep} | "
                   f"{r['route'] or ''} |")
    return "\n".join(out) + "\n\n"


LEGEND = """## How to read the table

Each model has one row per program version: `dump` is the raw network written as code, `agent_N` is
a Claude Code refactoring of it.

- **run**: `task/kind`; `hand` = handcrafted net, `seedN` = trained with that seed.
- **training**: how well the *network* learned the real task (accuracy for classifiers, R² for
  regression). "NOT learned" marks a training failure (kept, never replaced).
- **version**: which program the row is about.
- **check**: the program compared with the *network* (never the target).
  **PASS exact** = same answers everywhere, up to float rounding; **PASS snap** = differs only where
  allowed (below); **FAIL** = differs where it must not; **ERROR** = invalid program.
- **label**: how the check was done. **exhaustive** = every possible input tried (a proof for that
  domain); **empirical** = about 280k sampled inputs, concentrated near decision boundaries (strong
  testing, not proof).
- **max err**: regression only; the largest |program − network|. Snap allows up to 2% of the
  network's output range.
- **disagree outside / inside (zone)**: classifiers only. *Outside* = disagreements where the network
  was confident (must be 0 to pass); *inside* = disagreements in the network's "unsure zone" near its
  decision line (allowed). The % is the share of the input space inside that zone. The `dump` rows
  show large *inside* counts that are only float32-vs-float64 rounding ties exactly on the decision
  line, not real disagreements.
- **task in / out**: the program graded against the *true* answer, report only. *in* = inside the
  training range, *out* = up to 50% beyond it (generalization). Accuracy for classifiers (1 = perfect),
  max abs error for regression (0 = perfect). Blank *out* = discrete inputs, nothing outside.
- **recovered**: yes = the program is the true algorithm (perfect inside and outside the range).
- **nodes (% of dump)**: program size in syntax-tree nodes, and as a share of the dump.
- **audit**: Claude runs only. **ok** = the transcript shows no file access outside the blind folder;
  **OUTSIDE PATHS** = it touched outside files (such runs are excluded from headline numbers);
  **no transcript** = could not be audited.
- **Claude**: time taken and input+output tokens (input includes cached context).

"""

BEATS_INTRO = ("Runs where a refactored program is closer to the *true task* than the network it was "
               "decompiled from (better inside or outside the training range, and not worse on either). "
               "The refactorer never saw the target; this is the \"snapping recovers the intended "
               "algorithm\" effect.\n\n")


def _better(metric: str, new, old) -> int:
    """1 if new is better than old, -1 if worse, 0 if equal or unknown."""
    if new is None or old is None or isinstance(new, str) or isinstance(old, str):
        return 0
    eps = 1e-9 + 1e-6 * abs(old)
    if metric == "accuracy":
        return 1 if new > old + eps else -1 if new < old - eps else 0
    return 1 if new < old - eps else -1 if new > old + eps else 0


def flagged_reproduction_md(root: Path) -> str:
    """A secondary reproduction count that also uses audit-flagged agent attempts, with the reason they are flagged."""
    strict = [r for r in reproduction(root) if r["reproduces"] is not None]
    rows = [r for r in reproduction(root, valid_only=False) if r["reproduces"] is not None]
    if len(rows) == len(strict):
        return ""
    out = ["The counts above leave out agent attempts with outside-path audit flags. In these runs the flags come from "
           "the agents' own scratch scripts, written to the session scratchpad that parallel agents shared; the "
           "transcripts were reviewed by hand and no task information was read. Counting those attempts too "
           "(secondary, not the headline):\n"]
    for split, label in (("heldout", "held-out"), ("distill", "in-sample")):
        rs = [r for r in rows if r["split"] == split]
        if rs:
            out.append(f"- **{label}**: {sum(r['reproduces'] for r in rs)} of {len(rs)} reproduced; tool recovered "
                       f"the true formula in {sum(r['ast_recovered'] for r in rs)}, Claude in "
                       f"{sum(bool(r['agent_recovered']) for r in rs)}")
    return "\n".join(out) + "\n\n"


def _beats_the_net(root: Path) -> list[str]:
    out = []
    for rd in all_runs(root):
        res_path = rd / "results.json"
        if not is_step1(rd) or not res_path.exists():
            continue
        res = json.loads(res_path.read_text(encoding="utf-8"))
        dump_t = res["versions"]["dump"]["task"]
        metric = dump_t.get("metric")
        for name, v in res["versions"].items():
            if name in ("dump", "formula") or not v["check"].get("snap_pass"):   # formula.py is the target itself
                continue
            t = v["task"]
            cmp = [_better(metric, t.get(k), dump_t.get(k)) for k in ("in_range", "out_of_range")]
            if max(cmp) == 1 and min(cmp) >= 0:
                what = "accuracy" if metric == "accuracy" else "max error"
                parts = [f"{label} {_num(dump_t.get(k), '.4g')} → {_num(t.get(k), '.4g')}"
                         for label, k in (("inside", "in_range"), ("outside", "out_of_range"))
                         if t.get(k) is not None]
                flag = " (audit-flagged run)" if (v.get("meta") or {}).get("outside_paths") else ""
                out.append(f"- **{res['run']}** {name}: {what} {', '.join(parts)}"
                           f"{'; recovered the true algorithm' if t.get('recovered') else ''}{flag}")
    return out


def spike_gate(root: Path = RUNS) -> tuple[bool, list[str]]:
    lines, n_ok, n_rec = [], 0, 0
    for run in SPIKE_RUNS:
        path = Path(root) / run / "results.json"
        if not path.exists():
            lines.append(f"{run}: no results")
            continue
        vs = json.loads(path.read_text(encoding="utf-8"))["versions"]
        dump_nodes = (vs["dump"]["size"] or {})["nodes"]
        attempts = {k: v for k, v in vs.items()
                    if k.startswith("agent_") and not ((v.get("meta") or {}).get("outside_paths"))}
        ok = [k for k, v in attempts.items()
              if v["check"]["snap_pass"] and (v["size"] or {}).get("nodes", 10**9) <= 0.5 * dump_nodes]
        rec = [k for k, v in attempts.items() if v["task"]["recovered"]]
        n_ok += bool(ok)
        n_rec += bool(rec)
        lines.append(f"{run}: {len(attempts)} valid attempt(s); passes check and <=50% of dump nodes: "
                     f"{', '.join(ok) or 'none'}; recovered expected formula: {', '.join(rec) or 'none'}")
    passed = n_ok >= 2 and n_rec >= 1
    lines.append(f"gate {'PASSED' if passed else 'NOT passed'}: {n_ok}/3 runs with a small passing attempt "
                 f"(need 2), {n_rec}/3 with the formula recovered (need 1)")
    return passed, lines
