import numpy as np
import pytest

from experiments.tasks import get_task
from nndc.check import check_program
from nndc.compile_back import CompileError, compile_program
from nndc.spec import Spec


@pytest.mark.parametrize("task, body", [
    ("absdiff", "return abs(x0 - x1)"),
    ("median3", "return max(min(x0, x1), min(max(x0, x1), x2))"),
    ("tax", "return 0.1 * min(x0, 20) + 0.2 * min(max(x0 - 20, 0), 60) + 0.4 * max(x0 - 80, 0)"),
    ("window", "return abs(x0) < 0.5"),
    ("xor", "return x0 != x1"),
])
def test_compiled_net_matches_program_exactly(task, body):
    spec = Spec.from_json(get_task(task).spec.blind())
    src = f"def net({', '.join(spec.arg_names)}):\n    {body}\n"
    mlp, info = compile_program(src, spec)
    assert info["exact"] and check_program(src, mlp, spec).exact_pass


def test_decision_list_argmax_exact():
    spec = Spec.from_json(get_task("iris_petal").spec.blind())
    src = ("def net(x0, x1):\n    if x0 + x1 < 3.25:\n        return 0\n"
           "    if x0 + 1.125 * x1 < 6.8:\n        return 1\n    return 2\n")
    mlp, info = compile_program(src, spec)
    assert info["exact"] and check_program(src, mlp, spec).exact_pass


def test_argmax_call_with_named_intermediates():
    spec = Spec.from_json(get_task("iris_petal").spec.blind())
    src = ("def net(x0, x1):\n    s = x0 + 2 * x1\n    return argmax(4 - s, relu(s - 4), 0.5 * x1)\n")
    mlp, _ = compile_program(src, spec)
    assert check_program(src, mlp, spec).exact_pass


def test_branch_assignment_and_fallthrough():
    spec = Spec.from_json(get_task("absdiff").spec.blind())
    src = ("def net(x0, x1):\n    d = x0 - x1\n    if d < 0:\n        d = -d\n"
           "    if x0 > 2.5 or x1 > 2.5:\n        return d < 1\n    return d < 0.5\n")
    spec_b = Spec(task="", output="binary", inputs=spec.inputs)
    mlp, _ = compile_program(src, spec_b)
    assert check_program(src, mlp, spec_b).exact_pass


def test_regression_jump_uses_one_ramp():
    spec = Spec.from_json(get_task("tax").spec.blind())
    mlp, info = compile_program("def net(x0):\n    return 10 if x0 > 100 else 0\n", spec)
    assert info["ramps"] == 1 and not info["exact"]
    assert np.allclose(mlp.forward(np.array([[99.0], [101.0]], np.float32))[:, 0], [0, 10], atol=1e-3)


def test_regression_ifexp_between_expressions():
    spec = Spec.from_json(get_task("absdiff").spec.blind())
    mlp, _ = compile_program("def net(x0, x1):\n    return x0 if x1 > 0 else -x0\n", spec)
    X = np.array([[1.5, 1.0], [1.5, -1.0], [-2.0, 2.0]], np.float32)
    assert np.allclose(mlp.forward(X)[:, 0], [1.5, -1.5, -2.0], atol=1e-4)


def test_product_of_variables_rejected():
    spec = Spec.from_json(get_task("absdiff").spec.blind())
    with pytest.raises(CompileError):
        compile_program("def net(x0, x1):\n    return x0 * x1\n", spec)
