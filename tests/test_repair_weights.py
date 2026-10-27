import json

import numpy as np

from nndc.cli import main
from nndc.repair.metrics import decide
from nndc.repair.run import ARMS_W, evaluate_repair, faithful_program, run_repair_baselines
from nndc.repair.weights import finetune, lp_repair
from nndc.run import build_run
from tests.helpers import hand


def test_lp_repair_fixes_points_and_reports_infeasible():
    mlp, spec = hand("window")
    X = np.array([[0.6], [0.0], [-1.5]])
    y = np.array([1, 1, 0.])   # make 0.6 positive
    fixed = lp_repair(mlp, spec, X, y, 0.0)
    assert fixed is not None and (decide(spec, fixed.forward(X.astype(np.float32))) == y).all()
    X2 = np.array([[0.1], [0.1]])
    y2 = np.array([1, 0.])     # contradictory labels
    assert lp_repair(mlp, spec, X2, y2, 0.0) is None


def test_lp_repair_regression_within_tolerance():
    mlp, spec = hand("absdiff")
    X = np.random.default_rng(0).uniform(-3, 3, (50, 2))
    y = np.abs(X[:, 0] - X[:, 1]) + 0.3
    fixed = lp_repair(mlp, spec, X, y, 0.01)
    assert fixed is not None
    assert np.abs(fixed.forward(X.astype(np.float32))[:, 0] - y).max() <= 0.011


def test_finetune_moves_toward_labels():
    mlp, spec = hand("absdiff")
    X = np.random.default_rng(0).uniform(-3, 3, (200, 2))
    y = np.abs(X[:, 0] - X[:, 1]) + 1
    new = finetune(mlp, spec, X, y)
    assert np.abs(new.forward(X.astype(np.float32))[:, 0] - y).mean() < 0.5


def test_evaluate_repair_end_to_end(tmp_path):
    rd = build_run("xor", 0, tmp_path, bug=True)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return x0 != x1 or (x0 == 1 and x1 == 1)\n")
    assert faithful_program(rd).name == "agent_1.py"
    run_repair_baselines(rd)
    (rd / "fix_1.py").write_text("def net(x0, x1):\n    return x0 != x1\n")
    res = evaluate_repair(rd)
    assert res["fix_1"]["heldout"]["accuracy"] == 1.0 and res["fix_1"]["reinsertion"]["exact_pass"]
    assert res["fix_1"]["heldout"]["generalization"] is None   # whole domain shown
    assert set(ARMS_W) <= set(res) and "r_tree" in res
    assert json.loads((rd / "repair" / "results.json").read_text())["fix_1"]["nodes"]


def test_cli_repair_baselines(tmp_path, capsys):
    rd = build_run("window", 0, tmp_path, bug=True)
    assert main(["repair", "baselines", str(rd)]) == 0
    assert "w_ft" in capsys.readouterr().out and (rd / "repair" / "w_retrain.npz").exists()
