import numpy as np
from sklearn.tree import DecisionTreeRegressor

from nndc.baselines.common import net_samples
from nndc.baselines.tree import run_tree, tree_to_program
from nndc.cli import main
from nndc.lang import evaluate, parse_program
from nndc.run import build_run, evaluate_run
from tests.helpers import hand


def test_tree_program_matches_sklearn_predict():
    mlp, spec = hand("absdiff")
    X, _, dec = net_samples(spec, mlp, n=2000)
    t = DecisionTreeRegressor(max_leaf_nodes=16, random_state=0).fit(X, dec)
    prog = parse_program(tree_to_program(t, spec), spec.arg_names)
    assert np.allclose(evaluate(prog, X), t.predict(X))


def test_window_tree_passes_with_few_leaves(tmp_path):
    rd = build_run("window", None, tmp_path)
    info = run_tree(rd)
    assert info["passed"] and info["leaves"] <= 4
    assert evaluate_run(rd)["versions"]["tree"]["check"]["snap_pass"]


def test_cli_baseline_survives_a_bad_run(tmp_path, capsys):
    build_run("window", None, tmp_path)
    (tmp_path / "broken" / "seed0").mkdir(parents=True)
    (tmp_path / "broken" / "seed0" / "spec.json").write_text("{}")
    assert main(["baseline", "tree", "--all", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ERROR" in out and "window/hand" in out
