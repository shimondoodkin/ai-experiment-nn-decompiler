import json
from dataclasses import replace

import pytest

import experiments.tasks as tasks_mod
from nndc.cli import main
from nndc.run import build_run, evaluate_run


@pytest.fixture
def short_training(monkeypatch):
    monkeypatch.setitem(tasks_mod.TASKS, "absdiff", replace(tasks_mod.TASKS["absdiff"], steps=50))


def test_build_hand_run(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    assert rd == tmp_path / "absdiff" / "hand"
    for f in ["spec.json", "weights.npz", "dump.py", "results.json"]:
        assert (rd / f).exists()
    res = json.loads((rd / "results.json").read_text())
    assert res["versions"]["dump"]["check"]["exact_pass"]


def test_rerun_keeps_weights_and_agent_attempts(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    w_before = (rd / "weights.npz").read_bytes()
    build_run("absdiff", None, root=tmp_path)
    assert (rd / "weights.npz").read_bytes() == w_before and (rd / "agent_1.py").exists()
    res = json.loads((rd / "results.json").read_text())
    assert res["versions"]["agent_1"]["task"]["recovered"]


def test_retrain_refuses_with_agent_attempts(tmp_path, short_training):
    rd = build_run("absdiff", 0, root=tmp_path)
    assert (rd / "net.pt").exists() and json.loads((rd / "train_log.json").read_text())["seed"] == 0
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return 0\n")
    with pytest.raises(RuntimeError, match="agent"):
        build_run("absdiff", 0, root=tmp_path, retrain=True)


def test_counterexamples_saved_and_replayed(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return x0 - x1\n")
    evaluate_run(rd)
    assert json.loads((rd / "counterexamples.json").read_text())


def test_cli_check_exit_codes(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    good = tmp_path / "g.py"
    good.write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    bad = tmp_path / "b.py"
    bad.write_text("def net(x0, x1):\n    return x0\n")
    assert main(["check", str(good), "--run", str(rd)]) == 0
    assert main(["check", str(bad), "--run", str(rd)]) == 1


def test_non_utf8_attempt_is_reported_not_fatal(tmp_path):
    rd = build_run("absdiff", None, root=tmp_path)
    (rd / "agent_1.py").write_bytes(b"def net(x0, x1):\n    # a \x96 b\n    return abs(x0 - x1)\n")
    res = evaluate_run(rd)
    assert "UTF-8" in res["versions"]["agent_1"]["check"]["error"]
    build_run("absdiff", None, root=tmp_path)    # the run folder keeps working


def test_formula_only_for_sigmoid(tmp_path):
    from experiments.tasks import expected_program, get_task
    from nndc.run import load_run

    rd = build_run("window", 0, root=tmp_path / "s", act="sigmoid")
    assert (rd / "formula.py").read_text(encoding="utf-8") == expected_program(get_task("window"))
    res = json.loads((rd / "results.json").read_text())
    assert "formula" in res["versions"]
    assert load_run(rd)[1].acts == ["sigmoid"]
    relu = build_run("window", 0, root=tmp_path / "r")
    assert not (relu / "formula.py").exists()


def test_formula_cex_not_shared(tmp_path):
    rd = build_run("absdiff", 0, root=tmp_path, act="sigmoid")
    (rd / "counterexamples.json").unlink(missing_ok=True)
    res = evaluate_run(rd)
    key = lambda c: (c["inputs"]["x0"], c["inputs"]["x1"])   # noqa: E731
    formula = {key(c) for c in res["versions"]["formula"]["check"]["counterexamples"]}
    others = {key(c) for name, v in res["versions"].items() if name != "formula"
              for c in v["check"]["counterexamples"]}
    assert formula   # the sharp formula misses the rounded kink of the sigmoid net
    shared = rd / "counterexamples.json"
    stored = {(c["x0"], c["x1"]) for c in json.loads(shared.read_text())} if shared.exists() else set()
    assert not (formula - others) & stored


def test_sigmoid_skips_hand(tmp_path):
    with pytest.raises(ValueError):
        build_run("xor", None, root=tmp_path, act="sigmoid")
    with pytest.raises(ValueError):
        build_run("xor", 0, root=tmp_path, bug=True, act="sigmoid")


def test_act_mismatch_refused(tmp_path):
    rd = build_run("window", 0, root=tmp_path)            # a ReLU net
    with pytest.raises(ValueError, match="relu"):
        build_run("window", 0, root=tmp_path, act="sigmoid")
    assert not (rd / "formula.py").exists()
