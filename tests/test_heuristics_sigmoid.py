"""Sigmoid/tanh units: the ReLU-only heuristics leave them alone, and the tool still finds a passing program."""
import numpy as np

from nndc.ast_tool import refactor
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.bounds import prune_dead_and_linear
from nndc.heuristics.decision_list import decision_list
from nndc.heuristics.motifs import as_relu_sum
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import evaluate, from_dump


def _net(sizes, acts, seed=3):
    rng = np.random.default_rng(seed)
    return MLP([rng.normal(size=(o, i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])],
               [rng.normal(size=o).astype(np.float32) for o in sizes[1:]], acts)


def _spec(n_in, output, n_out=1):
    return Spec(task="", output=output, n_outputs=n_out,
                inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(n_in)])


def test_prune_keeps_sigmoid_units():
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    net = from_dump("def net(x0):\n    h1_0 = sigmoid(x0 - 5)\n    return 2 * h1_0 + 1\n", spec)
    pruned, _ = prune_dead_and_linear(net, spec)
    assert [(u.name, u.act) for u in pruned.units] == [("h1_0", "sigmoid")]
    X = np.linspace(0, 1, 101)[:, None]
    np.testing.assert_array_equal(evaluate(pruned, X), evaluate(net, X))


def test_relu_only_routes_skip_sigmoid():
    mlp = _net([2, 4, 1], ["sigmoid"])
    spec = _spec(2, "regression")
    assert as_relu_sum(from_dump(dump_program(mlp, spec), spec)) is None
    mlp = _net([2, 4, 3], ["sigmoid"])
    spec = _spec(2, "argmax", 3)
    assert decision_list(from_dump(dump_program(mlp, spec), spec), spec, mlp) is None


def test_refactor_sigmoid_net_passes():
    for sizes, acts, kind in [([2, 4, 1], ["sigmoid"], "regression"), ([2, 4, 3], ["tanh"], "argmax")]:
        mlp = _net(sizes, acts)
        spec = _spec(sizes[0], kind, sizes[-1])
        src, trace = refactor(dump_program(mlp, spec), spec, mlp)
        assert check_program(src, mlp, spec).snap_pass, (acts, src)
        if kind == "argmax":
            assert "decision-list: not applicable" in trace["trace_lines"]
