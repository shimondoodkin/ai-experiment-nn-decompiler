import numpy as np

from experiments.irrational_analysis import (formula_found, kinks_1d, merge_kinks, optimal_knots, pysr_found,
                                             won_on_size)
from nndc.model import MLP


def row(family, passed=True, terms=0, nodes=5):
    return {"family": family, "passed": passed, "terms": terms, "nodes": nodes}


def test_formula_found_needs_task_family():
    r = formula_found([row("sqrt", nodes=9), row("exp", passed=False)], "exp")
    assert not r["formula_found"] and r["other_families"] == ["sqrt"] and r["route_pick"] == "sqrt"
    r = formula_found([row("sqrt", nodes=9), row("exp", terms=2, nodes=12)], "exp")
    assert r["formula_found"] and r["found_terms"] == 2 and r["formula_nodes"] == 12 and r["route_pick"] == "sqrt"


def test_identity_never_counts():
    r = formula_found([row("identity", terms=3)], "exp")
    assert not r["formula_found"] and r["other_families"] == [] and r["route_pick"] == "identity"


def test_more_than_four_terms_is_not_found():
    assert not formula_found([row("exp", terms=5)], "exp")["formula_found"]


def test_merge_kinks_counts_stacked_units_once():
    assert merge_kinks([0.090, 0.093, 0.096, 1.2], width=4.0) == [0.093, 1.2]


def test_kinks_1d_relu_and_dead_units():
    W1 = np.array([[1.0], [1.0], [-1.0], [1.0]], np.float32)
    b1 = np.array([-0.5, 5.0, -5.0, 0.0], np.float32)     # kink 0.5; always on; never on; kink 0 with zero output weight
    mlp = MLP([W1, np.array([[1.0, 1.0, 1.0, 0.0]], np.float32)], [b1, np.zeros(1, np.float32)])
    k = kinks_1d(mlp, -2, 2)
    assert k["kinks"] == [0.5] and k["dead_units"] == 2


def test_optimal_knots_for_square_are_even():
    got = optimal_knots(lambda x: x * x, -2, 2, 4)
    assert np.allclose(got, [-1.5, -0.5, 0.5, 1.5], atol=0.05)


def test_won_on_size():
    assert won_on_size(7, 25, 58)
    assert not won_on_size(7, 6, 58)
    assert won_on_size(7, None, 58)          # 2-D: no hinge fit, ast4 only
    assert not won_on_size(None, 25, 58)


def test_pysr_found_by_family_function():
    assert pysr_found("def net(x0):\n    return (2.0 * exp(x0))\n", "exp")
    assert not pysr_found("def net(x0):\n    return (2.0 * x0)\n", "exp")
    assert pysr_found("def net(x0):\n    return (1.0 / (x0 + 1.0))\n", "reciprocal")
    assert pysr_found("def net(x0, x1):\n    return (x0 * x1)\n", "quadratic")
    assert pysr_found("def net(x0, x1):\n    return ((x0 * x0) + (x1 * x1) < 4.0)\n", "conic")
