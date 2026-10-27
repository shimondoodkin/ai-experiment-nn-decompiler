import importlib.util

import numpy as np
import pytest
import sympy

from nndc.baselines.pysr_run import UnsupportedExpr, run_pysr, snap_expr, sympy_to_program
from nndc.lang import evaluate, parse_program
from nndc.run import build_run, evaluate_run
from nndc.spec import InputSpec, Spec


def two_float_spec():
    return Spec(task="", output="regression",
                inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])


def test_printer_handles_piecewise_and_max():
    x0, x1 = sympy.symbols("x0 x1")
    e = sympy.Max(x0, x1) + sympy.Piecewise((1.0, x0 > 0.5), (0.0, True)) + 2 * sympy.Abs(x1) ** 2
    src = sympy_to_program(e, two_float_spec())
    prog = parse_program(src, ["x0", "x1"])
    X = np.array([[1.0, -2.0], [0.0, 3.0]])
    assert np.allclose(evaluate(prog, X), [1 + 1 + 8, 3 + 0 + 18])


def test_printer_relu_and_division_by_constant():
    x0, x1 = sympy.symbols("x0 x1")
    e = sympy.Piecewise((0.0, x0 < 0), (x0, True)) - x1 / 4
    prog = parse_program(sympy_to_program(e, two_float_spec()), ["x0", "x1"])
    assert np.allclose(evaluate(prog, np.array([[-1.0, 2.0], [2.0, 2.0]])), [-0.5, 1.5])


def test_printer_rejects_float_power():
    x0 = sympy.Symbol("x0")
    with pytest.raises(UnsupportedExpr):
        sympy_to_program(x0 ** 0.5, two_float_spec())


def test_printer_rejects_division_by_variable():
    x0, x1 = sympy.symbols("x0 x1")
    with pytest.raises(UnsupportedExpr):
        sympy_to_program(x0 / x1, two_float_spec())


def test_snap_expr_by_position():
    x0 = sympy.Symbol("x0")
    e = snap_expr(0.987 * x0 + 0.0031, out_range=6.0, in_range=6.0)
    assert e == x0   # 0.987 -> 1 (5% rel), 0.0031 -> 0 (1% of 6)


def test_snap_expr_threshold_uses_input_range():
    x0 = sympy.Symbol("x0")
    # 1% of the input range (1e-3) reaches 0.5; 1% of the output range (1e-4) would not
    e = snap_expr(sympy.Piecewise((1.0, x0 > 0.5004), (0.0, True)), out_range=0.01, in_range=0.1)
    assert sympy.Float(0.5004) not in e.atoms(sympy.Float)
    assert sympy.Float(0.5) in e.atoms(sympy.Float) or sympy.Rational(1, 2) in e.atoms(sympy.Rational)


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("pysr") is None, reason="needs uv run --group pysr")
def test_run_pysr_on_absdiff(tmp_path):
    rd = build_run("absdiff", None, tmp_path)
    info = run_pysr(rd)
    assert info["best"] and (rd / "pysr.py").exists()
    assert evaluate_run(rd)["versions"]["pysr"]["check"]["snap_pass"]
