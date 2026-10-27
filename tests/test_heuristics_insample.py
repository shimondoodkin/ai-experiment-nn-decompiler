"""Improvements made after the in-sample run (Task 8 step 1); synthetic nets only, never runs/."""
import numpy as np

from nndc.check import check_program, net_output
from nndc.dump import dump_program
from nndc.heuristics.boundary import boundary_expr
from nndc.heuristics.boundary_scan import scan_boundary_expr
from nndc.heuristics.cleanup import program_from_expr
from nndc.heuristics.motifs import as_relu_sum
from nndc.heuristics.spline import spline_expr
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import from_dump


def _mlp(layers):
    return MLP([np.array(W, np.float32) for W, _ in layers], [np.array(b, np.float32) for _, b in layers])


def test_boundary_direction_has_positive_leading_sign():
    # first unit looks along -x: the rule must still read abs(x0) < 0.5, without a derived variable
    mlp = _mlp([([[-0.98], [1.02]], [0, 0]), ([[-1, -1]], [0.51])])
    spec = Spec(task="", output="binary", inputs=[InputSpec("x0", "x0", "float", -2, 2)])
    expr, derived = boundary_expr(from_dump(dump_program(mlp, spec), spec), spec)
    assert derived == {} and expr == "abs(x0) < 0.5"


def test_spline_drops_small_hinge_at_the_domain_edge():
    # convex 0.1x + 0.1 relu(x-20) + 0.2 relu(x-80), plus a small concave hinge at 196 of 200 (fitting artifact)
    mlp = _mlp([([[1], [1], [1], [1]], [0, -20, -80, -196]), ([[0.1, 0.1, 0.2, -0.08]], [0])])
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 200)])
    expr = spline_expr(as_relu_sum(from_dump(dump_program(mlp, spec), spec)), spec)
    assert expr is not None and expr.startswith("max(")
    assert check_program(program_from_expr(expr, spec), mlp, spec).snap_pass


def test_scan_finds_band_despite_bias_units_in_other_directions():
    # 0.2 - |a-b| plus an always-on unit along another direction (a slight tilt the strict rule rejects)
    mlp = _mlp([([[1, -1], [-1, 1], [0.3, 0.1]], [0, 0, 5]), ([[-1, -1, 0.05]], [0.2 - 0.25])])
    spec = Spec(task="", output="binary",
                inputs=[InputSpec("x0", "x0", "float", -2, 2), InputSpec("x1", "x1", "float", -2, 2)])
    net = from_dump(dump_program(mlp, spec), spec)
    score = lambda X: net_output(mlp, spec, X)[1]  # noqa: E731
    r = scan_boundary_expr(score, spec, [{"x0": 1.0, "x1": -1.0}, {"x0": 0.3, "x1": 0.1}])
    assert r is not None
    expr, derived = r
    src = program_from_expr(expr, spec, derived)
    assert "abs(" in src and "0.2" in src and check_program(src, mlp, spec).snap_pass


def test_scan_snaps_to_measurement_precision_not_range():
    # band at 0.203 with large logits: should snap to 0.2, not the coarser 0.25 the full d-range would allow
    mlp = _mlp([([[1, -1], [-1, 1]], [0, 0]), ([[-50, -50]], [50 * 0.203])])
    spec = Spec(task="", output="binary",
                inputs=[InputSpec("x0", "x0", "float", -2, 2), InputSpec("x1", "x1", "float", -2, 2)])
    expr, derived = scan_boundary_expr(lambda X: net_output(mlp, spec, X)[1], spec, [{"x0": 1.0, "x1": -1.0}])
    assert expr == "abs(d0) < 0.2" and derived == {"d0": "x0 - x1"}


def test_tool_tries_spline_on_the_unsnapped_net():
    # convex spline with noise hinges near the left edge and a small hinge at the right edge (like an in-sample run)
    from nndc.ast_tool import refactor
    mlp = _mlp([([[1], [-1], [1], [-1], [1], [1]], [-2.0, 78.0, -2.05, 196.4, -0.89, -1.62]),
                ([[0.144, 0.216, 0.174, -0.086, -0.0125, 0.0085]], [-0.3955])])
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 200)])
    src, trace = refactor(dump_program(mlp, spec), spec, mlp)
    assert trace["winner"] == "spline" and src.strip().splitlines()[-1].strip().startswith("return max(")
