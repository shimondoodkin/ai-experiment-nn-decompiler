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


@pytest.mark.parametrize("src, kind", [
    ("def net(x0, x1):\n    return argmax(x0, x1)\n", "argmax"),                              # not units
    ("def net(x0, x1):\n    o0 = relu(x0)\n    o1 = relu(x1)\n    return argmax(o0, o1)\n", "argmax"),   # relu outputs
    ("def net(x0, x1):\n    o0 = x0\n    o1 = x1\n    return argmax(o0, o1)\n", "regression"),  # wrong output kind
    ("def net(x0, x1):\n    return x0 - x1\n", "argmax"),                                       # argmax spec, scalar return
])
def test_from_dump_rejects_malformed_outputs_with_value_error(src, kind):
    spec = Spec(task="", output=kind, n_outputs=2,
                inputs=[InputSpec("x0", "x0", "float", 0, 1), InputSpec("x1", "x1", "float", 0, 1)])
    with pytest.raises(ValueError, match="line"):
        from_dump(src, spec)


def test_roundtrip_sigmoid_tanh():
    from nndc.lang import evaluate as lang_eval, parse_program
    from nndc.model import MLP
    rng = np.random.default_rng(4)
    sizes = [2, 4, 4, 1]
    mlp = MLP([rng.normal(size=(o, i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])],
              [rng.normal(size=o).astype(np.float32) for o in sizes[1:]], ["sigmoid", "tanh"])
    spec = Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(2)])
    net = from_dump(dump_program(mlp, spec), spec)
    assert [u.act for u in net.units] == ["sigmoid"] * 4 + ["tanh"] * 4
    X = np.random.default_rng(5).uniform(-3, 3, (2000, 2))
    ref = mlp.forward(X.astype(np.float32))[:, 0]
    np.testing.assert_allclose(evaluate(net, X)[:, 0], ref, rtol=1e-5)
    np.testing.assert_allclose(lang_eval(parse_program(to_program(net), spec.arg_names), X), ref, rtol=1e-5)


def test_post_interval():
    import math
    from nndc.symnet import post_interval
    assert post_interval("relu", -2.0, 3.0) == (0.0, 3.0)
    assert post_interval("linear", -2.0, 3.0) == (-2.0, 3.0)
    assert post_interval("sigmoid", -2.0, 3.0) == pytest.approx((1 / (1 + math.exp(2)), 1 / (1 + math.exp(-3))))
    assert post_interval("tanh", -2.0, 3.0) == pytest.approx((math.tanh(-2.0), math.tanh(3.0)))
    assert post_interval("sigmoid", -1e4, 1e4) == (0.0, 1.0)
