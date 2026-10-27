import json

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
