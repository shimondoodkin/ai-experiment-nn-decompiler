"""Sigmoid/tanh units: the ReLU-only heuristics leave them alone, and the tool still finds a passing program."""
import numpy as np
import pytest

from nndc.ast_tool import refactor
from nndc.check import check_program
from nndc.dump import dump_program
from nndc.heuristics.bounds import prune_dead_and_linear
from nndc.heuristics.decision_list import _patterns, decision_list, fold
from nndc.heuristics.motifs import as_relu_sum
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from nndc.symnet import Affine, SymNet, Unit, evaluate, from_dump


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


@pytest.mark.parametrize("act", ["sigmoid", "tanh"])
def test_relu_only_routes_skip_sigmoid(act):
    mlp = _net([2, 4, 1], [act])
    spec = _spec(2, "regression")
    assert as_relu_sum(from_dump(dump_program(mlp, spec), spec)) is None
    mlp = _net([2, 4, 3], [act])
    spec = _spec(2, "argmax", 3)
    assert decision_list(from_dump(dump_program(mlp, spec), spec), spec, mlp) is None


def test_relu_only_routes_fail_closed_on_unknown_act():
    """An act that is neither relu nor linear (a future "step") is refused, not read as relu or linear."""
    step = SymNet(["x0", "x1"], [Unit("h1_0", 1, Affine({"x0": 1.0, "x1": -1.0}, 0.5), "step")],
                  [Affine({"h1_0": 2.0}, 0.0)], "regression")
    assert as_relu_sum(step) is None
    mlp = _net([2, 4, 3], ["relu"])
    spec = _spec(2, "argmax", 3)
    argmax_step = SymNet(["x0", "x1"], [Unit("h1_0", 1, Affine({"x0": 1.0}, 0.0), "step")],
                         [Affine({"h1_0": 1.0}, 0.0), Affine({"x1": 1.0}, 0.0), Affine({}, 0.0)], "argmax")
    assert decision_list(argmax_step, spec, mlp) is None
    X = np.zeros((1, 2))
    with pytest.raises(ValueError, match="step"):
        _patterns(argmax_step, X)
    with pytest.raises(ValueError, match="step"):
        fold(argmax_step, {"h1_0": True}, Affine({"h1_0": 1.0}, 0.0))


def test_refactor_sigmoid_net_passes():
    for sizes, acts, kind in [([2, 4, 1], ["sigmoid"], "regression"), ([2, 4, 3], ["tanh"], "argmax")]:
        mlp = _net(sizes, acts)
        spec = _spec(sizes[0], kind, sizes[-1])
        src, trace = refactor(dump_program(mlp, spec), spec, mlp)
        assert check_program(src, mlp, spec).snap_pass, (acts, src)
        if kind == "argmax":
            assert "decision-list: not applicable" in trace["trace_lines"]


# --- project B: sigmoid moves (docs/plans/2026-10-28-sigmoid-nets.md, Task 4 move list) ---

def _trained(task_name, act="sigmoid", seed=0):
    from experiments.tasks import get_task
    from experiments.train import train

    t = get_task(task_name)
    mlp, _ = train(t, seed, act=act)
    return mlp, Spec.from_json(t.spec.blind())


def _passes(src, mlp, spec):
    return check_program(src, mlp, spec).snap_pass


def test_step_clamp_program_matches_symnet():
    from nndc.lang import evaluate as lang_eval
    from nndc.lang import parse_program
    from nndc.symnet import to_program

    net = SymNet(["x0", "x1"], [
        Unit("h1", 1, Affine({"x0": 1.0, "x1": -0.3}, 0.1), "step"),
        Unit("h2", 1, Affine({"x0": 2.0, "x1": 1.0}, 0.0), "clamp"),
        Unit("h3", 2, Affine({"h1": 1.0, "h2": 1.0}, 0.0), "linear"),
    ], [Affine({"h3": 1.0, "h2": 0.5, "h1": -1.0}, 0.2)], "regression")
    X = np.random.default_rng(0).uniform(-3, 3, (1000, 2))
    src = to_program(net)
    np.testing.assert_allclose(lang_eval(parse_program(src, ["x0", "x1"]), X), evaluate(net, X)[:, 0], atol=1e-12)


def test_replace_units_on_window():
    from nndc.heuristics.sigmoid import replace_units
    from nndc.symnet import to_program

    mlp, spec = _trained("window")
    net, _ = prune_dead_and_linear(from_dump(dump_program(mlp, spec), spec), spec)
    out, trace = replace_units(net, spec, mlp)
    assert [u.act for u in out.units] == ["step"] * 4, trace
    assert _passes(to_program(out), mlp, spec)


def test_const_unit_inlined():
    from nndc.heuristics.sigmoid import replace_units
    from nndc.symnet import to_program

    mlp = MLP([np.array([[20.0], [0.01]], np.float32), np.array([[1.0, 1.0]], np.float32)],
              [np.array([0.0, 0.3], np.float32), np.zeros(1, np.float32)], ["sigmoid"])
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", -1, 1)])
    net = from_dump(dump_program(mlp, spec), spec)
    out, trace = replace_units(net, spec, mlp)
    assert any("const" in t for t in trace), trace
    pruned, _ = prune_dead_and_linear(out, spec)
    assert len(pruned.units) == 1 and _passes(to_program(pruned), mlp, spec)


def test_merge_steps_on_window():
    from nndc.heuristics.sigmoid import merge_steps, replace_units
    from nndc.symnet import to_program

    mlp, spec = _trained("window")
    net, _ = prune_dead_and_linear(from_dump(dump_program(mlp, spec), spec), spec)
    merged, _ = merge_steps(replace_units(net, spec, mlp)[0], spec, mlp)
    assert sum(u.act == "step" for u in merged.units) <= 3
    assert _passes(to_program(merged), mlp, spec)


def test_linear_scores_iris_petal():
    from nndc.heuristics.sigmoid import linear_scores

    mlp, spec = _trained("iris_petal")
    src = linear_scores(spec, mlp)
    assert src is not None and _passes(src, mlp, spec)


def test_linear_cascade_iris_petal():
    from nndc.heuristics.sigmoid import linear_cascade

    mlp, spec = _trained("iris_petal")
    src = linear_cascade(spec, mlp)
    assert src is not None and _passes(src, mlp, spec)


def test_knot_fit_tax_seed0():
    from pathlib import Path

    from nndc.heuristics.sigmoid import knot_fit
    from nndc.run import load_run

    spec, mlp = load_run(Path("runs_sigmoid/tax/seed0"))
    blind = Spec.from_json(spec.blind())
    src = knot_fit(blind, mlp)
    assert src is not None and _passes(src, mlp, blind)


def test_sigmoid_routes_skip_relu_nets():
    from tests.helpers import hand

    mlp, spec = hand("absdiff")
    _, trace = refactor(dump_program(mlp, spec), spec, mlp)
    routes = [c["route"] for c in trace["candidates"]]
    assert not [r for r in routes if r.startswith("sigmoid-") or r in ("linear-scores", "linear-cascade", "knot-fit")]
