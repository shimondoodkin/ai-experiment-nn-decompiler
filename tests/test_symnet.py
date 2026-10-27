import numpy as np
import pytest

from nndc.check import build_input_set, check_program
from nndc.dump import dump_program
from nndc.spec import InputSpec, Spec
from nndc.symnet import Affine, evaluate, from_dump, substitute, to_program, unit_bounds
from tests.helpers import hand


@pytest.mark.parametrize("name", ["absdiff", "deadredundant", "xor", "triangle", "window"])
def test_roundtrip_dump_symnet_program(name):
    mlp, spec = hand(name)
    net = from_dump(dump_program(mlp, spec), spec)
    X = build_input_set(mlp, spec)[0][:5000]
    np.testing.assert_allclose(evaluate(net, X)[:, 0], mlp.forward(X.astype(np.float32))[:, 0], atol=1e-5)
    assert check_program(to_program(net), mlp, spec).exact_pass


def test_unit_bounds_deadredundant():
    mlp, spec = hand("deadredundant")
    b = unit_bounds(from_dump(dump_program(mlp, spec), spec), spec)
    assert b["h1_0"] == pytest.approx((0, 2)) and b["h1_1"] == pytest.approx((-3, -1))


def test_from_dump_rejects_non_dump_shape():
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    with pytest.raises(ValueError, match="line"):
        from_dump("def net(x0):\n    if x0 > 1:\n        return 1\n    return 0\n", spec)


def test_substitute_inlines_affine():
    mlp, spec = hand("deadredundant")
    net = from_dump(dump_program(mlp, spec), spec)
    net2 = substitute(net, "h1_1", Affine({}, 0.0))
    assert "h1_1" not in to_program(net2)
