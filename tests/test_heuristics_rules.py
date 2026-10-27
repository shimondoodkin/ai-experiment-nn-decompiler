import numpy as np

from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.boundary import boundary_expr
from nndc.heuristics.cleanup import program_from_expr
from nndc.heuristics.fit import fit_expr
from nndc.heuristics.truthtable import truth_table_expr
from nndc.lang import evaluate, parse_program
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import from_dump
from tests.helpers import hand


def test_truth_table_finds_xor():
    mlp, spec = hand("xor")
    decide = lambda X: (mlp.forward(X.astype(np.float32))[:, 0] > 0).astype(float)  # noqa: E731
    assert truth_table_expr(decide, spec) in ("x0 != x1", "x1 != x0")


def test_truth_table_skips_large_or_continuous():
    _, spec = hand("absdiff")
    assert truth_table_expr(lambda X: X[:, 0] > 0, spec) is None


def test_boundary_snaps_window_from_trained_like_net():
    # perturbed window: 0.51 - relu(1.02 x) - relu(-0.98 x)
    mlp = MLP([np.array([[1.02], [-0.98]], np.float32), np.array([[-1, -1]], np.float32)],
              [np.zeros(2, np.float32), np.array([0.51], np.float32)])
    spec = Spec(task="", output="binary", inputs=[InputSpec("x0", "x0", "float", -2, 2)])
    expr, derived = boundary_expr(from_dump(dump_program(mlp, spec), spec), spec)
    src = program_from_expr(expr, spec, derived)
    assert "abs(" in src and "0.5" in src and check_program(src, mlp, spec).snap_pass


def test_fit_finds_median():
    spec = Spec(task="", output="regression",
                inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(3)])
    expr = fit_expr(lambda X: np.median(X, axis=1) * 1.003 + 0.001, spec)
    X = np.random.default_rng(1).uniform(-3, 3, (1000, 3))
    prog = parse_program(program_from_expr(expr, spec), spec.arg_names)
    np.testing.assert_allclose(evaluate(prog, X), np.median(X, axis=1), atol=1e-9)


def test_fit_rejects_poor_fits():
    spec = Spec(task="", output="regression",
                inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])
    assert fit_expr(lambda X: np.sin(3 * X[:, 0]) * X[:, 1], spec) is None
