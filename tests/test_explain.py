import json
from dataclasses import replace

import numpy as np
import pytest

from nndc.explain import describe_residual, explain_run, formula_centre
from nndc.model import MLP, save_npz
from nndc.spec import InputSpec, Spec

SPEC1 = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 4)])
SPEC2 = Spec(task="", output="regression",
             inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])


def test_residual_finds_edge_error():
    r = describe_residual(SPEC1, lambda X: X[:, 0] + 0.1 * np.maximum(0, X[:, 0] - 3.8), "def net(x0):\n    return x0\n")
    assert r["where"] == "edge" and len(r["by_slice"]) == 10
    assert r["max_err"] == pytest.approx(0.02 / 4.02, rel=1e-2)
    assert abs(r["by_slice"][-1]) > 0 and all(abs(x) < 1e-12 for x in r["by_slice"][:-1])


def test_residual_inside():
    r = describe_residual(SPEC1, lambda X: X[:, 0] + 0.1 * np.exp(-((X[:, 0] - 2) / 0.1) ** 2),
                          "def net(x0):\n    return x0\n")
    assert r["where"] == "inside"


def test_residual_by_sector_shows_ripple():
    def predict(X):
        r, t = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
        return r + 0.05 * np.cos(5 * t)
    res = describe_residual(SPEC2, predict, "def net(x0, x1):\n    return sqrt(x0 * x0 + x1 * x1)\n")
    assert len(res["by_ring"]) == 10 and len(res["by_sector"]) == 12
    assert res["centre"] == pytest.approx((0.0, 0.0), abs=1e-6)
    spread = max(res["by_sector"]) - min(res["by_sector"])
    assert spread > 3 * max(abs(x) for x in res["by_ring"] if x is not None)


def test_formula_centre_vertex_and_fallback():
    assert formula_centre("def net(x0, x1):\n    return sqrt((x0 - 0.5) * (x0 - 0.5) + x1 * x1)\n", SPEC2) == \
        pytest.approx((0.5, 0.0), abs=1e-6)
    assert formula_centre("def net(x0, x1):\n    return x0 * x1\n", SPEC2) == (0.0, 0.0)      # saddle: box centre
    assert formula_centre("def net(x0, x1):\n    return exp(0.5 * x0 + x1)\n", SPEC2) == (0.0, 0.0)  # linear u
    src = ("def net(x0, x1):\n    u = 2 * x0 * x0 + x1 * x1 - 4 * x0\n"
           "    return 0.5 * log(u + 9) + 0.1 * relu(u - 1)\n")
    assert formula_centre(src, SPEC2) == pytest.approx((1.0, 0.0), abs=1e-6)


def _sigmoid_run(tmp_path, units, spec):
    """A run folder by hand: blind spec, weights and dump of a one-layer sigmoid net (out, w..., b) per unit."""
    from nndc.dump import dump_program

    a = np.array(units, dtype=np.float64)
    n = len(spec.inputs)
    mlp = MLP([a[:, 1:1 + n].astype(np.float32), a[None, :, 0].astype(np.float32)],
              [a[:, 1 + n].astype(np.float32), np.zeros(1, np.float32)], ["sigmoid"])
    rd = tmp_path / "t" / "seed0"
    rd.mkdir(parents=True)
    spec.save(rd / "spec.json")
    save_npz(mlp, rd / "weights.npz")
    (rd / "dump.py").write_text(dump_program(mlp, spec), encoding="utf-8")
    return rd


def test_explain_run_writes_markdown(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0), (0.5, -3.0, 6.0), (1e-5, 1.0, 0.0)], SPEC1)
    near = {"family": "sqrt", "src": "def net(x0):\n    return sqrt(x0)\n", "err": 0.025, "terms": 0, "nodes": 3}
    (rd / "ast5_trace.json").write_text(json.dumps({"candidates": [], "winner": "dump", "near": near}))
    text = explain_run(rd)
    assert (rd / "explain.md").read_text(encoding="utf-8") == text
    assert "## Formula" in text and "## Units" in text and "## Groups" in text
    assert "sqrt(x0)" in text and "2.5%" in text and "negligible" in text
    assert "explain:" in (rd / "run_git.txt").read_text()


def test_explain_run_shows_the_passing_formula(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0)], SPEC1)
    src = "def net(x0):\n    return sigmoid(2 * x0 - 4)\n"
    cand = {"route": "family", "src": src, "nodes": 6, "check": {"snap_pass": True}}
    (rd / "ast5_trace.json").write_text(json.dumps({"candidates": [cand], "winner": "family", "near": None}))
    text = explain_run(rd)
    assert "passes the check" in text and "sigmoid(2 * x0 - 4)" in text


def test_explain_relu_net_two_zones(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 1.0, -2.0), (-1.0, 1.0, -1.0)], SPEC1)
    spec_relu = MLP([np.array([[1.0], [1.0]], np.float32), np.array([[1.0, -1.0]], np.float32)],
                    [np.array([-2.0, -1.0], np.float32), np.zeros(1, np.float32)])
    save_npz(spec_relu, rd / "weights.npz")
    text = explain_run(rd)
    assert "active" in text and "low knee" not in text
    assert "no ast5 trace" in text


def test_explain_two_layer_net_formula_only(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0)], SPEC1)
    two = MLP([np.ones((2, 1), np.float32), np.ones((2, 2), np.float32), np.ones((1, 2), np.float32)],
              [np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(1, np.float32)], ["sigmoid", "sigmoid"])
    save_npz(two, rd / "weights.npz")
    text = explain_run(rd)
    assert "## Formula" in text and "needs one hidden layer" in text and "## Groups" not in text


def test_explain_blind(tmp_path):
    spec = replace(SPEC1, task="secret_task_name")
    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0)], spec)
    assert "secret_task_name" not in explain_run(rd)


def test_cli_explain(tmp_path, capsys):
    from nndc.cli import main

    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0)], SPEC1)
    assert main(["explain", "--all", "--root", str(tmp_path)]) == 0
    assert (rd / "explain.md").exists() and "explain.md" in capsys.readouterr().out


def test_explain_faithful_fit_splits_distortion_and_ripple(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 0.3, -0.6)], SPEC1)          # nearly a line on [0, 4]
    near = {"family": "identity", "src": "def net(x0):\n    return 0.07 * x0 + 0.36\n", "err": 0.025, "terms": 0,
            "nodes": 5}
    (rd / "ast5_trace.json").write_text(json.dumps({"candidates": [], "winner": "dump", "near": near}))
    text = explain_run(rd)
    sec = text.split("## Faithful fit", 1)[1].split("## Units", 1)[0]
    assert "x0" in sec and "distortion" in sec and "ripple" in sec
    assert "| term | clean | faithful | effect |" in sec


def test_explain_faithful_fit_of_the_passing_family(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0)], SPEC1)
    src = "def net(x0):\n    return exp(x0)\n"
    cand = {"route": "family", "src": src, "nodes": 3, "check": {"snap_pass": True}}
    rows = [{"family": "log", "src": "def net(x0):\n    return log(x0)\n"}, {"family": "exp", "src": src}]
    (rd / "ast5_trace.json").write_text(json.dumps({"candidates": [cand], "winner": "family", "near": None,
                                                    "families": rows}))
    sec = explain_run(rd).split("## Faithful fit", 1)[1].split("## Units", 1)[0]
    assert "exp(" in sec and "log(" not in sec


def test_explain_faithful_conic(tmp_path):
    spec = replace(SPEC2, output="binary", threshold=0.5)
    rd = _sigmoid_run(tmp_path, [(1.0, 1.0, 0.0, 0.0)], spec)
    src = "def net(x0, x1):\n    return x0 * x0 + x1 * x1 < 4\n"
    cand = {"route": "conic", "src": src, "nodes": 18, "check": {"snap_pass": True}}
    row = {"family": "conic", "coefs": [0.98743, 1.0, 0.014847, -0.0072522, -0.00024445, 3.94251],
           "snapped": [1.0, 1.0, 0.0, 0.0, 0.0, 4.0], "src": src}
    (rd / "ast5_trace.json").write_text(json.dumps({"candidates": [cand], "winner": "conic", "near": None,
                                                    "families": [row]}))
    sec = explain_run(rd).split("## Faithful fit", 1)[1].split("## Units", 1)[0]
    assert "0.9874 * x0 * x0" in sec and "< 3.943" in sec
    assert "| x0² | 1 | 0.9874 |" in sec and "decide differently" in sec


def test_explain_no_faithful_without_formula(tmp_path):
    rd = _sigmoid_run(tmp_path, [(1.0, 2.0, -4.0)], SPEC1)
    (rd / "ast5_trace.json").write_text(json.dumps({"candidates": [], "winner": "dump", "near": None}))
    assert "## Faithful fit" not in explain_run(rd)
