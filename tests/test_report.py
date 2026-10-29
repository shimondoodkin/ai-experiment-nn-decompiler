import json

import pytest

from nndc.report import build_report, spike_gate, with_real_names
from nndc.run import build_run, evaluate_run
from nndc.spec import InputSpec, Spec


def test_real_names_substitution():
    spec = Spec(task="t", output="regression",
                inputs=[InputSpec("x0", "income", "float", 0, 1), InputSpec("x1", "rate", "float", 0, 1)])
    # whole tokens only, single pass: x10 is not an input and stays
    assert with_real_names("return x0 + x1 * x10", spec) == "return income + rate * x10"


def test_report_lists_versions(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    evaluate_run(rd)
    md = build_report(tmp_path)
    assert "absdiff/hand" in md and "agent_1" in md and "abs(a - b)" in md


def test_spike_gate(tmp_path):
    for t in ["xor", "absdiff", "neareq"]:
        rd = tmp_path / t / "seed0"
        rd.mkdir(parents=True)
        ok = t != "xor"
        (rd / "results.json").write_text(json.dumps({"versions": {
            "dump": {"size": {"nodes": 100}, "check": {"snap_pass": True}, "task": {"recovered": False}},
            "agent_1": {"size": {"nodes": 20 if ok else 90}, "check": {"snap_pass": True},
                        "task": {"recovered": t == "absdiff"}}}}))
    passed, lines = spike_gate(tmp_path)
    assert passed and len(lines) >= 3


def test_spike_gate_ignores_audit_flagged_attempts(tmp_path):
    for t in ["xor", "absdiff", "neareq"]:
        rd = tmp_path / t / "seed0"
        rd.mkdir(parents=True)
        (rd / "results.json").write_text(json.dumps({"versions": {
            "dump": {"size": {"nodes": 100}, "check": {"snap_pass": True}, "task": {"recovered": False}},
            "agent_1": {"size": {"nodes": 20}, "check": {"snap_pass": True}, "task": {"recovered": True},
                        "meta": {"outside_paths": ["C:/secret"]}}}}))
    passed, _ = spike_gate(tmp_path)
    assert not passed


def test_agent_notes_headings_are_nested_under_the_run_section(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    (rd / "agent_1_notes.md").write_text("# Notes\n## What I changed\nx\n## Heuristics\n- y\n")
    evaluate_run(rd)
    md = build_report(tmp_path)
    assert "\n#### Heuristics" in md and "\n## Heuristics" not in md and "\n### Notes" in md


def test_report_explains_columns_and_lists_programs_that_beat_the_net(tmp_path):
    rd = build_run("absdiff_perturbed", None, root=tmp_path)    # the perturbed net is slightly off abs(a-b)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    evaluate_run(rd)
    md = build_report(tmp_path)
    assert "## How to read the table" in md and "**disagree outside / inside (zone)**" in md
    beats = md.split("## Where the refactored program beats the network")[1].split("\n## ")[0]
    assert "absdiff_perturbed/hand" in beats and "agent_1" in beats


def test_spike_gate_counts_only_claude_attempts(tmp_path):
    for t in ["xor", "absdiff", "neareq"]:
        rd = tmp_path / t / "seed0"
        rd.mkdir(parents=True)
        (rd / "results.json").write_text(json.dumps({"versions": {
            "dump": {"size": {"nodes": 100}, "check": {"snap_pass": True}, "task": {"recovered": False}},
            "ast": {"size": {"nodes": 10}, "check": {"snap_pass": True}, "task": {"recovered": True}}}}))
    passed, _ = spike_gate(tmp_path)
    assert not passed


def _fake_run(root, task, seed, versions, learned=True):
    from experiments.tasks import get_task

    rd = root / task / f"seed{seed}"
    rd.mkdir(parents=True)
    get_task(task).spec.save(rd / "spec.json")
    (rd / "results.json").write_text(json.dumps({
        "task": task, "run": f"{task}/seed{seed}", "kind": "trained",
        "train": {"quality": {"r2": 0.99}, "learned": learned}, "versions": versions}))
    return rd


def _v(snap=True, exact=False, recovered=False, nodes=10, err=0.05, margin=0.1, cex=()):
    return {"check": {"exact_pass": exact, "snap_pass": snap, "error": None, "max_abs_err": err, "margin": margin,
                      "disagreements_outside": 0, "disagreements_inside": 0, "zone_fraction": 0.0,
                      "counterexamples": list(cex), "label": "empirical"},
            "task": {"recovered": recovered, "in_range": 0.0, "out_of_range": 0.0}, "size": {"nodes": nodes}, "meta": None}


def test_formula_md_rows(tmp_path):
    from nndc.report import formula_md

    cex = [{"inputs": {"x0": 0.0, "x1": 0.0}, "net": 0.6, "program": 0.0, "in_zone": False}]
    _fake_run(tmp_path, "absdiff", 0, {"dump": _v(nodes=200), "agent_1": _v(nodes=30), "ast": _v(nodes=40),
                                       "formula": _v(snap=False, recovered=True, nodes=5, err=0.6, cex=cex)})
    _fake_run(tmp_path, "iris", 0, {"dump": _v(nodes=400)})
    md = formula_md(tmp_path)
    assert md.count("| absdiff/seed0 |") == 1 and "iris/seed0" not in md
    assert "0.6" in md and "FAIL" in md   # formula verdict and its counterexample's net value


def test_activation_compare_md_pairs_runs(tmp_path):
    from nndc.report import activation_compare_md

    relu, sig = tmp_path / "relu", tmp_path / "sig"
    _fake_run(relu, "absdiff", 0, {"dump": _v(), "agent_1": _v(recovered=True, nodes=7), "ast": _v(nodes=9)})
    _fake_run(sig, "absdiff", 0, {"dump": _v(), "agent_1": _v(nodes=30), "ast": _v(nodes=235)})
    _fake_run(sig, "xor", 0, {"dump": _v()})
    md = activation_compare_md(relu, sig)
    assert md.count("| absdiff/seed0 |") == 1 and "xor/seed0" not in md and "235" in md


@pytest.mark.slow
def test_default_report_unchanged():
    from pathlib import Path

    assert build_report(Path("runs")) == Path("REPORT.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def test_beats_skips_formula(tmp_path):
    from nndc.report import _beats_the_net

    dump = _v(nodes=200)
    dump["task"].update({"metric": "max_abs_err", "in_range": 0.5, "out_of_range": 0.5})
    formula = _v(nodes=5)
    formula["task"].update({"metric": "max_abs_err", "in_range": 0.0, "out_of_range": 0.0, "recovered": True})
    _fake_run(tmp_path, "absdiff", 0, {"dump": dump, "formula": formula})
    assert not [line for line in _beats_the_net(tmp_path) if "formula" in line]


def test_flagged_reproduction_note(tmp_path):
    from nndc.report import flagged_reproduction_md

    flagged = _v(recovered=True, nodes=9)
    flagged["meta"] = {"outside_paths": ["C:/scratch/an.py"], "transcript_found": True}
    rd = _fake_run(tmp_path, "absdiff", 0, {"dump": _v(nodes=200), "agent_1": flagged, "ast": _v(recovered=True, nodes=10)})
    (rd / "ast_trace.json").write_text(json.dumps({"winner": "structural-snap"}))
    md = flagged_reproduction_md(tmp_path)
    assert "outside" in md and "1 of 1" in md      # the flagged attempt is counted in the secondary count


def _kink_files(rd, m=4):
    tr = {"winner": "kink-merge", "kink_merge": {
        "applies": None,
        "per_offset": [{"off": 0.02, "kinks": 128, "lines": 32, "cancelled": 14},
                       {"off": 0.1, "kinks": 128, "lines": 16, "cancelled": 11}],
        "winner": {"off": 0.02, "m": m, "tol": 0.02,
                   "lines": [{"normal": {"x0": 1.0, "x1": -1.0}, "t": -0.25, "strength": 1.0}] * m}}}
    (rd / "ast2_trace.json").write_text(json.dumps(tr))
    (rd / "ast2.py").write_text("def net(x0, x1):\n    return relu(x0 - x1 + 0.25)\n")
    (rd / "formula.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")


def test_kink_md_rows(tmp_path):
    from nndc.report import kink_md

    for seed in (0, 3):
        rd = _fake_run(tmp_path, "absdiff", seed, {"dump": _v(nodes=200), "agent_1": _v(nodes=28),
                                                   "ast2": _v(nodes=30), "formula": _v(snap=False, recovered=True)})
        _kink_files(rd)
    md = kink_md(tmp_path)
    assert md.count("| absdiff/seed0 |") == 1 and md.count("| absdiff/seed3 |") == 1
    assert "| development |" in md and "| held-out |" in md
    assert "128 → 32 → 14 → 4" in md
    assert "max(a - b + 0.25, 0)" in md                     # kept lines in real input names
    assert "relu(a - b + 0.25)" in md and "abs(a - b)" in md  # ast2 next to formula.py


def test_kink_md_empty(tmp_path):
    from nndc.report import kink_md

    _fake_run(tmp_path, "absdiff", 0, {"dump": _v(), "ast": _v()})
    assert kink_md(tmp_path) == ""


def test_kink_md_marks_lost_candidates_and_other_runs(tmp_path):
    from nndc.report import kink_md

    won = _fake_run(tmp_path, "absdiff", 0, {"dump": _v(), "ast2": _v(nodes=30)})
    _kink_files(won)
    lost = _fake_run(tmp_path, "absdiff", 1, {"dump": _v(), "ast2": _v(nodes=22)})
    _kink_files(lost)
    tr = json.loads((lost / "ast2_trace.json").read_text())
    tr["winner"] = "knot-fit"
    (lost / "ast2_trace.json").write_text(json.dumps(tr))
    iris = _fake_run(tmp_path, "iris", 3, {"dump": _v(), "ast2": _v()})
    (iris / "ast2_trace.json").write_text(json.dumps({"winner": "linear-scores", "kink_merge": {"applies": "argmax output"}}))
    rows = {ln.split(" | ")[0][2:]: ln for ln in kink_md(tmp_path).splitlines() if ln.startswith("| ") and "seed" in ln}
    assert "candidate only:" in rows["absdiff/seed1"] and "candidate only:" not in rows["absdiff/seed0"]
    assert "iris/seed3" not in rows          # the route does not apply there: not part of this section


def test_report_reads_newest_tool_version(tmp_path):
    from nndc.report import activation_compare_md, formula_md

    relu, sig = tmp_path / "relu", tmp_path / "sig"
    _fake_run(relu, "absdiff", 0, {"dump": _v(), "ast": _v(nodes=9)})
    _fake_run(sig, "absdiff", 0, {"dump": _v(), "ast2": _v(nodes=77),
                                  "formula": _v(snap=False, recovered=True, nodes=5)})
    assert "77 nodes" in formula_md(sig)
    md = activation_compare_md(relu, sig)
    assert "9 nodes" in md and "77 nodes" in md


def test_run_section_shows_ast2(tmp_path):
    rd = build_run("window", 0, root=tmp_path, act="sigmoid")
    (rd / "agent_1.py").write_text("def net(x0):\n    return 0.5 - abs(x0)\n")
    (rd / "ast2.py").write_text("def net(x0):\n    return 0.25 - abs(x0) / 2\n")
    evaluate_run(rd)
    md = build_report(tmp_path, compare_to=tmp_path)
    assert "AST tool (no LLM, `ast2`)" in md and "0.25 - abs(x) / 2" in md


# --- ast3 ------------------------------------------------------------------------------------------------------

def _tool_files(rd, name, winner="kink-merge"):
    tr = {"winner": winner, "kink_merge": {
        "applies": None, "per_offset": [{"off": 0.02, "kinks": 128, "lines": 32, "cancelled": 14}],
        "winner": {"off": 0.02, "m": 1, "tol": 0.0, "lines": [{"normal": {"x0": 1.0, "x1": -1.0}, "t": 0.0, "strength": 1.0}]}}}
    (rd / f"{name}_trace.json").write_text(json.dumps(tr))
    (rd / f"{name}.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")


def test_newest_is_ast3(tmp_path):
    from nndc.report import formula_md

    _fake_run(tmp_path, "absdiff", 0, {"dump": _v(), "ast2": _v(nodes=77), "ast3": _v(nodes=55),
                                       "formula": _v(snap=False, recovered=True, nodes=5)})
    md = formula_md(tmp_path)
    assert "55 nodes" in md and "77 nodes" not in md


def test_ast2_vs_ast3_worse_rule(tmp_path):
    from nndc.report import ast_compare_rows, kink_md

    cases = {0: (_v(nodes=80), _v(snap=False, nodes=60)),    # ast3 fails: worse
             1: (_v(nodes=80), _v(nodes=90)),                # ast3 more nodes: worse
             2: (_v(nodes=80), _v(nodes=70))}                # ast3 fewer nodes: not worse
    for seed, (a2, a3) in cases.items():
        rd = _fake_run(tmp_path, "absdiff", seed, {"dump": _v(), "ast2": a2, "ast3": a3})
        _tool_files(rd, "ast2")
        _tool_files(rd, "ast3")
    rows = {r["run"]: r["worse"] for r in ast_compare_rows(tmp_path)}
    assert rows == {"absdiff/seed0": True, "absdiff/seed1": True, "absdiff/seed2": False}
    assert "ast3 is worse on 2 of 3 runs" in kink_md(tmp_path)


def test_heldout_json_split(tmp_path):
    from nndc.report import kink_md, newest_reproduction_md

    (tmp_path / "heldout.json").write_text(json.dumps({"version": "ast3", "freeze": "abc1234",
                                                       "runs": ["absdiff/seed3"]}))
    for seed in (0, 3):
        rd = _fake_run(tmp_path, "absdiff", seed, {"dump": _v(), "agent_1": _v(nodes=8), "ast3": _v(nodes=9)})
        _tool_files(rd, "ast3")
    rows = {ln.split(" | ")[0][2:]: ln for ln in kink_md(tmp_path).splitlines() if ln.startswith("| ") and "/seed" in ln}
    assert "| held-out |" in rows["absdiff/seed3"] and "| development |" in rows["absdiff/seed0"]
    md = newest_reproduction_md(tmp_path)
    assert "ast3" in md and "abc1234" in md and "held-out" in md


def test_compare_heading_other_root(tmp_path):
    from nndc.report import activation_compare_md

    a, b = tmp_path / "runs_sigmoid", tmp_path / "runs_tanh"
    _fake_run(a, "absdiff", 0, {"dump": _v(), "ast3": _v(nodes=9)})
    _fake_run(b, "absdiff", 0, {"dump": _v(), "ast3": _v(nodes=11)})
    md = activation_compare_md(a, b)
    assert "runs_sigmoid" in md and "ReLU" not in md and "11 nodes" in md


# --- ast4 ------------------------------------------------------------------------------------------------------

def test_newest_is_ast4(tmp_path):
    from nndc.report import formula_md, kink_md

    rd = _fake_run(tmp_path, "absdiff", 0, {"dump": _v(), "ast3": _v(nodes=55), "ast4": _v(nodes=44),
                                            "formula": _v(snap=False, recovered=True, nodes=5)})
    _tool_files(rd, "ast3")
    _tool_files(rd, "ast4")
    md = formula_md(tmp_path)
    assert "44 nodes" in md and "55 nodes" not in md
    assert "## Kink merge (ast4)" in kink_md(tmp_path)


def test_ast4_compared_with_each_older_version(tmp_path):
    from nndc.report import ast_compare_rows, kink_md

    cases = {0: (_v(nodes=80), _v(nodes=90), _v(nodes=80)),   # ast4 = ast2, smaller than ast3
             1: (_v(nodes=80), _v(nodes=70), _v(nodes=75))}   # ast4 worse than ast3 only
    for seed, (a2, a3, a4) in cases.items():
        rd = _fake_run(tmp_path, "absdiff", seed, {"dump": _v(), "ast2": a2, "ast3": a3, "ast4": a4})
        for name in ("ast2", "ast3", "ast4"):
            _tool_files(rd, name)
    assert {r["run"]: r["worse"] for r in ast_compare_rows(tmp_path, "ast2", "ast4")} == \
        {"absdiff/seed0": False, "absdiff/seed1": False}
    assert {r["run"]: r["worse"] for r in ast_compare_rows(tmp_path, "ast3", "ast4")} == \
        {"absdiff/seed0": False, "absdiff/seed1": True}
    md = kink_md(tmp_path)
    assert "### ast2 vs ast4" in md and "### ast3 vs ast4" in md and "ast2 vs ast3" not in md
    assert "ast4 is worse on 0 of 2 runs" in md and "ast4 is worse on 1 of 2 runs" in md


def test_run_section_shows_ast4(tmp_path):
    rd = build_run("window", 0, root=tmp_path, act="sigmoid")
    (rd / "agent_1.py").write_text("def net(x0):\n    return 0.5 - abs(x0)\n")
    (rd / "ast3.py").write_text("def net(x0):\n    return 0.25 - abs(x0) / 2\n")
    (rd / "ast4.py").write_text("def net(x0):\n    return 0.75 - abs(x0) / 2\n")
    evaluate_run(rd)
    md = build_report(tmp_path, compare_to=tmp_path)
    assert "AST tool (no LLM, `ast4`)" in md and "0.75 - abs(x) / 2" in md
