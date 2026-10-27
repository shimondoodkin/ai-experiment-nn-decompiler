from pathlib import Path

import numpy as np
import pytest

from nndc.repair.data import repair_data
from nndc.repair.metrics import added_tests, decide, repair_metrics
from nndc.run import build_run
from nndc.spec import InputSpec, Spec


def test_metrics_counts():
    spec = Spec(task="", output="binary", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    y = np.array([1, 1, 0, 0, 1.])
    orig = np.array([1, 0, 0, 1, 1.])   # wrong on 1 and 3
    new = np.array([1, 1, 1, 0, 1.])    # fixes 1, 3; breaks 2
    m = repair_metrics(spec, orig, new, y, 0.0, whole_domain=False)
    assert m["generalization"] == 1.0 and m["efficacy"] == 1.0 and m["drawdown"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(0.8) and m["n_failing"] == 2 and m["n_right"] == 3


def test_regression_metrics_use_tolerance():
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    y = np.array([0.0, 1.0, 2.0])
    m = repair_metrics(spec, np.array([0.5, 1.0, 2.0]), np.array([0.05, 1.0, 2.3]), y, 0.1, whole_domain=False)
    assert m["generalization"] == 1.0 and m["drawdown"] == pytest.approx(0.5) and m["max_abs_err"] == pytest.approx(0.3)


def test_whole_domain_generalization_is_none(tmp_path):
    rd = build_run("xor", None, tmp_path)
    d = repair_data(rd)
    assert d.whole_domain and len(d.X_heldout) == 4
    m = repair_metrics(Spec.load(rd / "spec.json"), d.y_heldout, d.y_heldout, d.y_heldout, 0.0, d.whole_domain)
    assert m["generalization"] is None


def test_formula_run_data_sizes(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    d = repair_data(rd)
    assert len(d.X_repair) == 2000 and len(d.X_heldout) == 20_000 and len(d.X_oor) == 20_000
    assert np.allclose(d.y_repair, np.abs(d.X_repair[:, 0] - d.X_repair[:, 1]))
    assert not d.flowers and not d.whole_domain


def test_iris_split_is_stratified_and_disjoint():
    d = repair_data(Path("runs/iris_petal/seed0"))
    assert len(d.X_repair) == 100 and len(d.X_heldout) == 50 and d.flowers and d.X_oor is None
    assert sorted(np.bincount(d.y_heldout.astype(int)).tolist()) == [16, 17, 17]


def test_decide_net_logits_and_programs():
    spec = Spec(task="", output="argmax", n_outputs=3, inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    assert decide(spec, np.array([[0.1, 0.5, 0.2], [2.0, 0.0, 1.0]])).tolist() == [1.0, 0.0]
    assert decide(spec, np.array([2.0, 1.0])).tolist() == [2.0, 1.0]


def test_added_test_support_flag():
    X = np.linspace(0, 1, 101)[:, None]
    orig = "def net(x0):\n    return x0 < 0.5\n"
    new = "def net(x0):\n    if x0 > 0.995:\n        return 1\n    return x0 < 0.5\n"
    rows = added_tests(orig, new, X, ["x0"])
    assert rows == [{"test": "x0 > 0.995", "support": 1, "flagged": True}]


def test_added_tests_see_through_named_values():
    X = np.linspace(-1, 1, 201)[:, None]
    orig = "def net(x0):\n    s = 2 * x0\n    return s < 0.5\n"
    new = "def net(x0):\n    s = 2 * x0\n    if s > -1:\n        return s < 0.5\n    return 1\n"
    rows = added_tests(orig, new, X, ["x0"])
    assert len(rows) == 1 and rows[0]["support"] == 51 and not rows[0]["flagged"]   # x0 <= -0.5: 51 points


def test_added_tests_not_flagged_on_tiny_domains():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    rows = added_tests("def net(x0, x1):\n    return x0 < 0.5\n", "def net(x0, x1):\n    return x0 != x1\n",
                       X, ["x0", "x1"])
    assert rows and not any(r["flagged"] for r in rows)
