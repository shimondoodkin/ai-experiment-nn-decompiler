import numpy as np

from experiments.irrational_analysis import (agent_fields, conic_shown, formula_found, kinks_1d, merge_kinks,
                                             optimal_knots, pysr_found, reproduced, won_on_size)
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


def test_conic_shown_is_the_kept_program():
    clean = "def net(x0, x1):\n    return x0 * x0 + x1 * x1 < 4\n"
    rows = [row("sqrt", passed=False),
            {"family": "conic", "passed": True, "snapped": [1, 1, 0, 0, 0, 4.05], "src": clean}]
    assert conic_shown(rows) == "`x0 * x0 + x1 * x1 < 4`"
    rows[1].update(passed=False, src="def net(x0, x1):\n    return 0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95\n")
    assert conic_shown(rows) == "`0.25 * x0 * x0 + x1 * x1 - 0.25 * x0 < 0.95` (fails)"
    assert conic_shown([row("sqrt")]) is None
    assert conic_shown([{"family": "conic", "passed": False, "src": None}]) is None


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


def test_near_fields():
    from experiments.irrational_analysis import near_fields

    near = {"family": "sqrt", "src": "s", "err": 0.025, "terms": 1, "nodes": 9}
    assert near_fields(near, "sqrt") == {"near_family": "sqrt", "near_err": 0.025, "near_terms": 1,
                                         "near_family_is_task_family": True}
    assert near_fields(near, "exp")["near_family_is_task_family"] is False
    assert near_fields(None, "exp") == {"near_family": None, "near_err": None, "near_terms": None,
                                        "near_family_is_task_family": None}


def test_reproduced_rules():
    assert reproduced(True, True, 41, 55, False, False) is False         # ast5 larger
    assert reproduced(True, True, 43, 35, True, False) is False          # the agent wrote the family, ast5 did not
    assert reproduced(True, True, 43, 35, False, False) is True
    assert reproduced(True, True, 43, 43, True, True) is True            # equal size counts
    assert reproduced(True, False, 43, 10, True, True) is False          # ast5 fails


def _res(agent_pass=True, agent_nodes=20, ast5_pass=True, ast5_nodes=10):
    return {"agent_1": {"check": {"snap_pass": agent_pass}, "size": {"nodes": agent_nodes}},
            "ast5": {"check": {"snap_pass": ast5_pass}, "size": {"nodes": ast5_nodes}}}


def test_agent_fields():
    a = agent_fields(_res(), "def net(x0):\n    return exp(x0)\n", "def net(x0):\n    return exp(x0)\n", "exp", False)
    assert a == {"agent_pass": True, "agent_nodes": 20, "agent_formula": True, "ast5_formula": True,
                 "reproduced": True}
    none = dict.fromkeys(a)
    assert agent_fields(_res(), "def net(x0):\n    return x0\n", "def net(x0):\n    return x0\n", "exp", True) == none
    assert agent_fields({"ast5": _res()["ast5"]}, None, "def net(x0):\n    return x0\n", "exp", False) == none


def test_family_check_ignores_comments():
    src = "def net(x0):\n    # its shape follows sqrt(x0) closely\n    return min(0.5 * x0 + 0.5, x0)  # not sqrt(x0)\n"
    assert not pysr_found(src, "sqrt")
    assert pysr_found("def net(x0):\n    # a line\n    return sqrt(x0)\n", "sqrt")
