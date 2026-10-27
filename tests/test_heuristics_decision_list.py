"""P10: argmax as a decision list, on synthetic nets only (never runs/)."""
import numpy as np

from nndc.ast_tool import refactor
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.decision_list import decision_list, fold
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import Affine, from_dump


def _mlp(layers):
    return MLP([np.array(W, np.float32) for W, _ in layers], [np.array(b, np.float32) for _, b in layers])


def _spec(*inputs):
    return Spec(task="", output="argmax", n_outputs=3, inputs=[InputSpec(f"x{k}", f"x{k}", t, lo, hi)
                                                               for k, (t, lo, hi) in enumerate(inputs)])


# two always-on units h0 = x0 + 1, h1 = x1 + 1; o0 = 2 - 2 x0, o1 = 0, o2 = x0 + x1 - 3:
# class 0 when x0 < 1, class 2 when x0 + x1 > 3, else class 1 (classes 0 and 2 never touch)
LINES = _mlp([([[1, 0], [0, 1]], [1, 1]), ([[-2, 0], [0, 0], [1, 1]], [4, 0, -5])])
LINES_SPEC = _spec(("float", 0, 4), ("float", 0, 2))


def test_fold_through_pattern():
    net = from_dump(dump_program(LINES, LINES_SPEC), LINES_SPEC)
    s = net.outputs[2].plus(net.outputs[1].scaled(-1.0))
    got = fold(net, {u.name: True for u in net.units}, s)
    assert got.coef == {"x0": 1.0, "x1": 1.0} and got.const == -3.0
    assert fold(net, {u.name: False for u in net.units}, s) == Affine({}, -5.0)


def test_two_linear_tests_pass_exact():
    net = from_dump(dump_program(LINES, LINES_SPEC), LINES_SPEC)
    src, snapped, trace = decision_list(net, LINES_SPEC, LINES)
    assert src.count("if ") == 2 and "argmax" not in src and "relu" not in src
    assert "x0 + x1 > 3" in src and ("x0 < 1" in src or "x0 > 1" in src)
    assert check_program(src, LINES, LINES_SPEC).exact_pass


def test_tool_picks_decision_list_for_argmax():
    src, trace = refactor(dump_program(LINES, LINES_SPEC), LINES_SPEC, LINES)
    assert trace["winner"].startswith("decision-list"), trace["winner"]
    assert check_program(src, LINES, LINES_SPEC).snap_pass


def test_bool_inputs_use_truth_table_tests():
    # o_k = k*s - k^2/2 with s = x0 + x1: argmax is the count of true inputs
    mlp = _mlp([([[1, 0], [0, 1]], [0, 0]), ([[0, 0], [1, 1], [2, 2]], [0, -0.5, -2])])
    spec = _spec(("bool", 0, 1), ("bool", 0, 1))
    net = from_dump(dump_program(mlp, spec), spec)
    src, _, trace = decision_list(net, spec, mlp)
    assert "relu" not in src and "argmax" not in src and check_program(src, mlp, spec).exact_pass, src
    assert any("by truth-table" in t for t in trace), trace


def test_snapped_versions_round_noisy_constants():
    noisy = _mlp([([[1, 0], [0, 1]], [1, 1]), ([[-2.01, 0], [0, 0], [1, 1.004]], [4.003, 0, -5.01])])
    net = from_dump(dump_program(noisy, LINES_SPEC), LINES_SPEC)
    _, snapped, _ = decision_list(net, LINES_SPEC, noisy)
    assert snapped and "x0 < 1\n" in snapped[0].replace(":", "\n") and "x0 + x1 > 3" in snapped[0], snapped
