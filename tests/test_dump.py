from pathlib import Path

import numpy as np
import pytest

from nndc.dump import dump_program
from nndc.lang import evaluate, parse_program
from nndc.model import MLP
from nndc.run import load_run
from nndc.spec import InputSpec, Spec


def random_mlp(sizes, seed):
    rng = np.random.default_rng(seed)
    return MLP([rng.normal(size=(o, i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])],
               [rng.normal(size=o).astype(np.float32) for o in sizes[1:]])


@pytest.mark.parametrize("sizes, kind, acts", [
    ([2, 3, 1], "binary", None), ([3, 8, 8, 1], "regression", None), ([2, 4, 3], "argmax", None),
    ([2, 3, 1], "binary", ["sigmoid"]), ([2, 3, 1], "binary", ["tanh"]),
    ([3, 8, 8, 1], "regression", ["relu", "sigmoid"]),
    ([2, 4, 3], "argmax", ["sigmoid"]), ([2, 4, 3], "argmax", ["tanh"])])
def test_dump_evaluates_like_net(sizes, kind, acts):
    mlp = random_mlp(sizes, seed=1)
    mlp.acts = acts or mlp.acts
    spec = Spec(task="secret_task", output=kind, n_outputs=sizes[-1],
                inputs=[InputSpec(f"x{i}", f"real{i}", "float", -3, 3) for i in range(sizes[0])])
    src = dump_program(mlp, spec)
    X = np.random.default_rng(2).uniform(-3, 3, (2000, sizes[0]))
    out = evaluate(parse_program(src, [f"x{i}" for i in range(sizes[0])]), X)
    ref = mlp.forward(X.astype(np.float32))
    if kind == "argmax":
        assert (out == ref.argmax(1)).mean() > 0.999
    else:
        np.testing.assert_allclose(out, ref[:, 0], rtol=1e-4, atol=1e-5)
    assert "secret_task" not in src and "real0" not in src


def test_zero_weights_omitted():
    mlp = MLP([np.array([[1.0, 0.0]], np.float32), np.array([[2.0]], np.float32)],
              [np.zeros(1, np.float32), np.zeros(1, np.float32)])
    spec = Spec(task="t", output="regression",
                inputs=[InputSpec("x0", "a", "float", 0, 1), InputSpec("x1", "b", "float", 0, 1)])
    src = dump_program(mlp, spec)
    body = src.split("def net(x0, x1):\n")[1]
    assert "x1" not in body                       # x1 appears only in the signature
    assert "h1_0 = relu(1 * x0)" in body


def test_dump_relu_text_unchanged():
    rd = Path("runs/absdiff/hand")
    spec, mlp = load_run(rd)
    committed = (rd / "dump.py").read_bytes().decode().replace("\r\n", "\n")
    assert dump_program(mlp, spec) == committed


def test_dump_writes_activation_name():
    mlp = random_mlp([2, 3, 1], seed=1)
    mlp.acts = ["tanh"]
    spec = Spec(task="t", output="regression", inputs=[InputSpec(f"x{i}", "a", "float", -3, 3) for i in range(2)])
    src = dump_program(mlp, spec)
    assert "h1_0 = tanh(" in src and "relu" not in src
