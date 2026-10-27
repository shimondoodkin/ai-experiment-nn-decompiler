import numpy as np

from nndc.baselines.common import net_samples, smallest_passing
from nndc.run import build_run, evaluate_run
from tests.helpers import hand


def test_net_samples_shapes_and_grid():
    mlp, spec = hand("xor")          # bool domain of 4 points -> full grid
    X, raw, dec = net_samples(spec, mlp, n=50)
    assert X.shape == (4, 2) and set(dec) <= {0.0, 1.0}
    mlp, spec = hand("absdiff")
    X, raw, dec = net_samples(spec, mlp, n=1000)
    assert X.shape == (1000, 2) and np.allclose(dec, np.abs(X[:, 0] - X[:, 1]), atol=1e-5)


def test_smallest_passing_prefers_fewest_nodes():
    mlp, spec = hand("absdiff")
    big = "def net(x0, x1):\n    return relu(x0 - x1) + relu(x1 - x0)\n"
    small = "def net(x0, x1):\n    return abs(x0 - x1)\n"
    wrong = "def net(x0, x1):\n    return x0\n"
    i, rows = smallest_passing([big, wrong, small], mlp, spec)
    assert i == 2 and rows[1]["snap_pass"] is False


def test_smallest_passing_none_when_nothing_passes():
    mlp, spec = hand("absdiff")
    i, rows = smallest_passing(["def net(x0, x1):\n    return x0\n", "not python("], mlp, spec)
    assert i is None and rows[1]["error"]


def test_evaluate_run_picks_up_baseline_versions(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    (rd / "tree.py").write_text("def net(x0, x1):\n    return abs(x0 - x1)\n")
    (rd / "io_agent_1.py").write_text("def net(x0, x1):\n    return x0\n")
    (rd / "fix_1.py").write_text("def net(x0, x1):\n    return x0\n")
    res = evaluate_run(rd)
    assert res["versions"]["tree"]["check"]["exact_pass"] and "io_agent_1" in res["versions"]
    assert "fix_1" not in res["versions"]
