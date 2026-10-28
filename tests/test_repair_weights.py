import json
from pathlib import Path

import numpy as np
import pytest

from nndc.cli import main
from nndc.model import MLP
from nndc.repair.metrics import decide
from nndc.repair.run import ARMS_W, evaluate_repair, faithful_program, run_repair_baselines
from nndc.repair.weights import _hidden, finetune, lp_repair
from nndc.run import build_run
from nndc.spec import Spec
from tests.helpers import hand, random_mlp


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


def test_lp_repair_never_returns_a_net_that_misses_its_points():
    # xor/seed2's net outputs ~1e-7: LP margins on that scale vanish in float32, so the result must be checked
    from nndc.run import load_run
    spec, mlp = load_run(Path("runs/xor/seed2"))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0.])
    fixed = lp_repair(mlp, spec, X, y, 0.0)
    assert fixed is None or (decide(spec, fixed.forward(X.astype(np.float32))) == y).all()


def test_reinsertion_reports_task_match_and_box_mismatch(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    (rd / "agent_1.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    (rd / "fix_1.py").write_text("def net(x0, x1):\n    return 5 if x0 > 0 else abs(x0 - x1)\n")   # a jump: one ramp
    res = evaluate_repair(rd)
    rein = res["fix_1"]["reinsertion"]
    assert rein["ramps"] == 1 and rein["task_match"] == 1.0 and 0 <= rein["box_mismatch"] < 0.001


def test_lp_repair_keeps_acts():
    from nndc.model import MLP
    from nndc.spec import InputSpec

    rng = np.random.default_rng(3)
    mlp = MLP([rng.normal(size=(6, 2)).astype(np.float32), rng.normal(size=(1, 6)).astype(np.float32)],
              [rng.normal(size=6).astype(np.float32), rng.normal(size=1).astype(np.float32)], ["sigmoid"])
    spec = Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", -2, 2) for i in range(2)])
    X = rng.uniform(-2, 2, (20, 2))
    y = mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)
    fixed = lp_repair(mlp, spec, X, y, 0.05)
    assert fixed is not None and fixed.acts == mlp.acts == ["sigmoid"]


def test_hidden_matches_old_relu():
    mlp = random_mlp([3, 7, 5, 2], seed=5)
    X = np.random.default_rng(6).uniform(-3, 3, (200, 3))
    np.testing.assert_array_equal(
        _hidden(mlp, X), np.maximum(mlp.preactivations(X)[-2].astype(np.float64), 0.0))
    np.testing.assert_array_equal(
        mlp.hidden(X), np.maximum(mlp.preactivations(X)[-2].astype(np.float64), 0.0))


def _mlp_with_acts(acts):
    rng = np.random.default_rng(0)
    sizes = [2] + [3] * len(acts) + [1]
    return MLP([rng.normal(size=(o, i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])],
               [rng.normal(size=o).astype(np.float32) for o in sizes[1:]], acts)


def test_retrain_arm_keeps_the_nets_activation(tmp_path, monkeypatch):
    import experiments.train as train_mod
    from nndc.repair.run import _retrain_clean

    seen = []
    monkeypatch.setattr(train_mod, "train", lambda task, seed, target=None, act="relu": seen.append((task.name, seed, act)) or ("net", {}))
    rd = tmp_path / "xor" / "seed3"
    rd.mkdir(parents=True)
    from experiments.tasks import get_task
    get_task("xor").spec.save(rd / "spec.json")
    assert _retrain_clean(rd, _mlp_with_acts(["sigmoid"])) == "net"
    assert _retrain_clean(rd, _mlp_with_acts(["tanh", "tanh"])) == "net"
    assert _retrain_clean(rd, _mlp_with_acts([])) == "net"
    assert [a for _, _, a in seen] == ["sigmoid", "tanh", "relu"] and {s for _, s, _ in seen} == {3}
    with pytest.raises(ValueError, match="one activation"):
        _retrain_clean(rd, _mlp_with_acts(["relu", "sigmoid"]))
