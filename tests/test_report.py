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
