import numpy as np
import pytest

from nndc.lang import ProgramError, evaluate, parse_program, program_size

X2 = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])


def ev(src, X=X2):
    return evaluate(parse_program(src), X)


def test_arith_and_functions():
    np.testing.assert_allclose(ev("def net(x0, x1):\n    return relu(x0 - x1) + abs(x1) / 2"), [1.0, 4.5, 0.0])
    np.testing.assert_allclose(ev("def net(x0, x1):\n    return max(x0, x1, 0.5) + sum((x0, x1))"), [5.0, 5.0, 0.5])
    np.testing.assert_allclose(ev("def net(x0, x1):\n    return argmax(x0, x1)"), [1, 0, 0])


def test_if_elif_else_per_row():
    src = ("def net(x0, x1):\n    if x0 > 2:\n        return 10\n    elif x1 > 1:\n"
           "        y = x1 * 3\n        return y\n    else:\n        return -1")
    np.testing.assert_allclose(ev(src), [6.0, 10.0, -1.0])


def test_bool_results_and_ifexp():
    np.testing.assert_allclose(ev("def net(x0, x1):\n    return abs(x0 - x1) < 1.5"), [1, 0, 1])
    np.testing.assert_allclose(ev("def net(x0, x1):\n    return 1 if x0 != x1 and not x0 > 2 else 0"), [1, 0, 0])


@pytest.mark.parametrize("src, word", [
    ("import math\ndef net(x0):\n    return x0", "Import"),
    ("def net(x0):\n    return math.floor(x0)", "Attribute"),
    ("def net(x0):\n    return x0 ** 2", "Pow"),
    ("def net(x0):\n    return sum([x0 for _ in (1, 2)])", "ListComp"),
    ("def f(x0):\n    return x0\ndef net(x0):\n    return f(x0)", "one function"),
    ("def net(x0):\n    for i in (1,2):\n        pass\n    return x0", "For"),
    ("def net(x0):\n    return x0 / 0", "division"),
    ("def net(x0):\n    return round(x0)", "round"),
])
def test_rejects_with_named_construct(src, word):
    with pytest.raises(ProgramError, match=word):
        parse_program(src)


def test_error_has_line_number():
    with pytest.raises(ProgramError, match="line 3"):
        parse_program("def net(x0):\n    y = x0\n    return y ** 2")


def test_signature_mismatch():
    with pytest.raises(ProgramError, match="arguments"):
        parse_program("def net(a, b):\n    return a", expected_args=["x0", "x1"])


def test_missing_return_on_some_rows():
    with pytest.raises(ProgramError, match="no return"):
        ev("def net(x0, x1):\n    if x0 > 2:\n        return 1")


def test_unassigned_name():
    with pytest.raises(ProgramError, match="h9"):
        ev("def net(x0, x1):\n    return h9")


def test_size():
    s = program_size(parse_program(
        "def net(x0, x1):\n    h = relu(x0 - 0.5)\n    if h > 1:\n        return h\n    return 0"))
    assert s["relus"] == 1 and s["ifs"] == 1 and s["constants"] == 3 and s["lines"] == 4


def test_read_program_handles_bom_utf16_and_rejects_bad_bytes(tmp_path):
    from nndc.lang import read_program
    src = "def net(x0):\n    return x0  # \u2013 dash\n"
    (tmp_path / "a.py").write_bytes(src.encode("utf-8-sig"))
    (tmp_path / "b.py").write_bytes(src.encode("utf-16"))
    (tmp_path / "c.py").write_bytes(b"def net(x0):\n    return x0  # \x96\n")
    assert read_program(tmp_path / "a.py") == src
    assert read_program(tmp_path / "b.py") == src
    with pytest.raises(ProgramError, match="UTF-8"):
        read_program(tmp_path / "c.py")


def test_sigmoid_tanh_eval():
    x = np.array([-1e4, -1.0, 0.0, 1.0, 1e4])
    got = ev("def net(x0):\n    return sigmoid(x0) + tanh(x0)", x[:, None])
    want = 0.5 * (1 + np.tanh(x / 2)) + np.tanh(x)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert program_size(parse_program("def net(x0):\n    return sigmoid(x0)"))["relus"] == 0
    from nndc.model import activate
    np.testing.assert_array_equal(ev("def net(x0):\n    return sigmoid(x0)", x[:, None]), activate("sigmoid", x))
    np.testing.assert_array_equal(ev("def net(x0):\n    return tanh(x0)", x[:, None]), activate("tanh", x))


def test_sigmoid_one_arg():
    with pytest.raises(ProgramError, match="exactly one argument"):
        parse_program("def net(x0, x1):\n    return sigmoid(x0, x1)")
    with pytest.raises(ProgramError, match="exactly one argument"):
        parse_program("def net(x0, x1):\n    return tanh(x0, x1)")


def test_sigmoid_no_overflow():
    from nndc.lang import _sigmoid
    x = np.array([-1e4, -1.0, 0.0, 1.0, 1e4])
    with np.errstate(over="raise", invalid="raise", divide="raise"):   # under is allowed: exp(-1e4) underflows
        got = _sigmoid(x)
    np.testing.assert_allclose(got, [0.0, 1 / (1 + np.e), 0.5, 1 / (1 + np.exp(-1.0)), 1.0], atol=1e-15)


def test_evaluator_sigmoid_goes_through_the_stable_function(monkeypatch):
    # evaluate() runs under np.errstate(all="ignore"), so an overflow in a naive formula would be silent:
    # check instead that the evaluator calls the stable _sigmoid (test_sigmoid_no_overflow covers that one)
    import nndc.lang as lang
    monkeypatch.setattr(lang, "_sigmoid", lambda z: np.full_like(z, 0.25))
    np.testing.assert_array_equal(ev("def net(x0):\n    return sigmoid(x0)", np.array([[-1e4], [0.0], [1e4]])), 0.25)


def test_exp_log_sqrt_evaluate():
    p = parse_program("def net(x0):\n    return exp(x0) + log(x0) + sqrt(x0)\n", ["x0"])
    X = np.array([[1.0], [4.0]])
    assert np.allclose(evaluate(p, X), np.exp(X[:, 0]) + np.log(X[:, 0]) + np.sqrt(X[:, 0]))


def test_division_by_expression():
    p = parse_program("def net(x0, x1):\n    return x0 / (x1 + 1)\n", ["x0", "x1"])
    assert np.allclose(evaluate(p, np.array([[3.0, 2.0]])), [1.0])


def test_new_functions_take_one_argument():
    for name in ("exp", "log", "sqrt"):
        with pytest.raises(ProgramError, match="exactly one argument"):
            parse_program(f"def net(x0):\n    return {name}(x0, x0)\n", ["x0"])


def test_out_of_domain_gives_nonfinite_without_warning():
    import warnings

    p = parse_program("def net(x0):\n    return log(x0) + sqrt(x0) + 1 / x0\n", ["x0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = evaluate(p, np.array([[-1.0], [0.0]]))
    assert not np.isfinite(out).any()


def test_new_function_counts_one_node():
    assert program_size(parse_program("def net(x0):\n    return exp(x0)\n"))["nodes"] == \
           program_size(parse_program("def net(x0):\n    return relu(x0)\n"))["nodes"]


def test_allowlist_mentions_new_functions():
    from nndc.lang import allowlist_summary

    s = allowlist_summary()
    assert "exp(x)" in s and "log(x)" in s and "sqrt(x)" in s and "division by a nonzero number constant" not in s
