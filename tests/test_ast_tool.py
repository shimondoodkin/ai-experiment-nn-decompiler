import numpy as np
import pytest

from experiments.grade import grade_task
from experiments.tasks import get_task
from nndc.ast_tool import refactor
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from tests.helpers import random_mlp


@pytest.mark.parametrize("name", ["absdiff", "absdiff_perturbed", "deadredundant", "max", "ramp", "triangle",
                                  "window", "neareq", "xor", "sparse8", "sum8"])
def test_refactor_handcrafted_recovers(name):
    t = get_task(name)
    mlp = t.handcrafted()
    spec = Spec.from_json(t.spec.blind())
    src, trace = refactor(dump_program(mlp, spec), spec, mlp)
    assert check_program(src, mlp, spec).snap_pass and grade_task(src, t)["recovered"], (name, src)
    assert trace["winner"] and trace["candidates"]


def test_refactor_requires_blind_spec():
    t = get_task("absdiff")
    mlp = t.handcrafted()
    with pytest.raises(ValueError, match="blind"):
        refactor(dump_program(mlp, t.spec), t.spec, mlp)


def test_two_layer_and_argmax_nets_do_not_crash():
    for sizes, kind in [([3, 6, 6, 1], "regression"), ([2, 4, 3], "argmax")]:
        mlp = random_mlp(sizes, seed=3)
        spec = Spec(task="", output=kind, n_outputs=sizes[-1],
                    inputs=[InputSpec(f"x{i}", f"x{i}", "float", -1, 1) for i in range(sizes[0])])
        src, trace = refactor(dump_program(mlp, spec), spec, mlp)
        assert check_program(src, mlp, spec).snap_pass


def _x0_net(n_in):
    """relu(x0) - relu(-x0) = x0, as a ReLU net with n_in inputs."""
    W0 = np.zeros((2, n_in), np.float32)
    W0[0, 0], W0[1, 0] = 1, -1
    return MLP([W0, np.array([[1, -1]], np.float32)], [np.zeros(2, np.float32), np.zeros(1, np.float32)])


def _spec(n_in, output="regression"):
    return Spec(task="", output=output, inputs=[InputSpec(f"x{i}", f"x{i}", "float", -1, 1) for i in range(n_in)])


NEAR_ROWS = [{"family": "sqrt", "near": [{"terms": 0, "src": "s0", "err": 0.025, "nodes": 9}]},
             {"family": "exp", "near": [{"terms": 1, "src": "e1", "err": 0.021, "nodes": 12}]}]


def test_ast5_trace_near_when_no_formula_passes(monkeypatch):
    import nndc.ast_tool as at

    monkeypatch.setattr(at, "family_candidates", lambda spec, mlp: ([], [dict(r) for r in NEAR_ROWS]))
    mlp, spec = _x0_net(1), _spec(1)
    _, trace = refactor(dump_program(mlp, spec), spec, mlp, version="ast5")
    assert trace["near"] == {"family": "sqrt", "src": "s0", "err": 0.025, "terms": 0, "nodes": 9}


def test_ast5_trace_near_none_when_formula_passes(monkeypatch):
    import nndc.ast_tool as at

    src = "def net(x0):\n    return x0\n"
    monkeypatch.setattr(at, "family_candidates", lambda spec, mlp: ([("identity", src)], [dict(r) for r in NEAR_ROWS]))
    mlp, spec = _x0_net(1), _spec(1)
    _, trace = refactor(dump_program(mlp, spec), spec, mlp, version="ast5")
    assert trace["near"] is None
    _, t4 = refactor(dump_program(mlp, spec), spec, mlp, version="ast4")
    assert "near" not in t4


def test_ast5_trace_near_is_the_failing_conic(monkeypatch):
    import nndc.ast_tool as at

    wrong = "def net(x0, x1):\n    return x1\n"     # the net decides x0 > 0
    monkeypatch.setattr(at, "conic_candidate", lambda spec, mlp: (wrong, {"family": "conic", "passed": False}))
    mlp, spec = _x0_net(2), _spec(2, "binary")
    _, trace = refactor(dump_program(mlp, spec), spec, mlp, version="ast5")
    near = trace["near"]
    assert near["family"] == "conic" and near["src"] == wrong and near["terms"] == 0
    assert 0.3 < near["err"] < 0.7
