import numpy as np
import pytest

from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.cleanup import program_from_expr
from nndc.heuristics.motifs import as_relu_sum, motif_expr
from nndc.heuristics.spline import spline_expr
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import from_dump
from tests.helpers import hand, normalize_ws, random_mlp


@pytest.mark.parametrize("name, expected", [
    ("absdiff", "abs(x0 - x1)"), ("max", "max(x0, x1)"), ("ramp", "min(max(x0, 0), 1)"),
    ("triangle", "max(0, 1 - abs(x0 - 3))"), ("window", "0.5 - abs(x0)"), ("neareq", "0.2 - abs(x0 - x1)"),
    ("sum8", "max(0, 0.02 * sum(x0, x1, x2, x3, x4, x5, x6, x7) - 20)"),
])
def test_motifs_on_handcrafted(name, expected):
    mlp, spec = hand(name)
    rs = as_relu_sum(from_dump(dump_program(mlp, spec), spec))
    src = program_from_expr(motif_expr(rs), spec)
    assert check_program(src, mlp, spec).exact_pass
    assert normalize_ws(src.strip().splitlines()[-1]) == f"return {expected}"


def test_spline_convex_to_max_of_lines():
    # 1-D convex net: 0.1*x + 0.1*relu(x-20) + 0.2*relu(x-80) on [0, 200]
    mlp = MLP([np.array([[1], [1], [1]], np.float32), np.array([[0.1, 0.1, 0.2]], np.float32)],
              [np.array([0, -20, -80], np.float32), np.zeros(1, np.float32)])
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 200)])
    expr = spline_expr(as_relu_sum(from_dump(dump_program(mlp, spec), spec)), spec)
    assert expr.startswith("max(") and check_program(program_from_expr(expr, spec), mlp, spec).snap_pass


def test_two_layer_net_has_no_relu_sum():
    mlp = random_mlp([3, 4, 4, 1], seed=0)
    spec = Spec(task="", output="regression",
                inputs=[InputSpec(f"x{i}", f"x{i}", "float", -1, 1) for i in range(3)])
    assert as_relu_sum(from_dump(dump_program(mlp, spec), spec)) is None
