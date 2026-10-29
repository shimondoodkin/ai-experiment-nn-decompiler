"""Kink merge: sigmoids expanded into ReLU kinks, kinks merged by line (nndc/heuristics/kinks.py)."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from experiments.tasks import uniform_sample
from nndc.check import check_program
from nndc.heuristics.kinks import (Kink, RealFit, evaluate_expansion, expand, fit_program, kink_merge, merge_lines,
                                     nearly_cancelled, refine, snap_fit, to_real, unit_box)
from nndc.model import MLP, activate
from nndc.run import load_run
from nndc.spec import InputSpec, Spec


def _blind(rd):
    spec, mlp = load_run(Path(rd))
    return Spec.from_json(spec.blind()), mlp


def chord_reference(mlp, spec, U, pieces):
    """The net with each sigmoid replaced by its chords on `pieces` uniform knots over the box's z range."""
    lo, span = unit_box(spec)
    W, b = mlp.weights[0].astype(np.float64), mlp.biases[0].astype(np.float64)
    Wu, bu = W * span, W @ lo + b
    corners = np.array(list(itertools.product([0, 1], repeat=len(lo))), float)
    H = []
    for j in range(len(b)):
        zc = corners @ Wu[j] + bu[j]
        t = np.linspace(zc.min(), zc.max(), pieces + 1)
        H.append(np.interp(U @ Wu[j] + bu[j], t, activate("sigmoid", t)))
    return np.stack(H, 1) @ mlp.weights[1][0].astype(np.float64) + float(mlp.biases[1][0])


def _spec2(lo=-3, hi=3):
    return Spec(task="", output="regression", inputs=[InputSpec(f"x{i}", f"x{i}", "float", lo, hi) for i in range(2)])


def test_expand_matches_chords():
    mlp = MLP([np.array([[2., -1.], [-1.5, 0.5], [0., 3.]], np.float32), np.array([[1., -2., 0.5]], np.float32)],
              [np.array([0.3, -0.2, 0.1], np.float32), np.array([0.4], np.float32)], ["sigmoid"])
    spec = _spec2()
    e = expand(spec, mlp, pieces=17)
    U = np.random.default_rng(0).uniform(0, 1, (5000, 2))
    np.testing.assert_allclose(evaluate_expansion(e, U), chord_reference(mlp, spec, U, 17), atol=1e-9)
    assert e.kinks
    assert all(k.normal[np.flatnonzero(np.abs(k.normal) > 1e-9)[0]] > 0 for k in e.kinks)
    assert all(abs(np.linalg.norm(k.normal) - 1) < 1e-12 for k in e.kinks)


def test_box_folding():
    # interior knots always cross the box; only a unit with a constant pre-activation has none
    spec = _spec2()
    U = np.random.default_rng(1).uniform(0, 1, (3000, 2))
    mlp0 = MLP([np.array([[1., 1.], [0., 0.]], np.float32), np.array([[1., 2.]], np.float32)],
               [np.array([0., 0.7], np.float32), np.array([0.], np.float32)], ["sigmoid"])
    e0 = expand(spec, mlp0, pieces=5)
    assert len(e0.kinks) == 4     # only unit 0 bends; the constant unit folds into the affine part
    np.testing.assert_allclose(evaluate_expansion(e0, U), chord_reference(mlp0, spec, U, 5), atol=1e-9)


def test_expansion_close_to_net_absdiff():
    blind, mlp = _blind("runs_sigmoid/absdiff/seed0")
    lo, span = unit_box(blind)
    X = uniform_sample(blind, np.random.default_rng(1), 50_000)
    net = mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)
    err = np.abs(evaluate_expansion(expand(blind, mlp), (X - lo) / span) - net).max()
    assert err <= 0.02 * np.ptp(net)


def test_merge_absdiff_lines_parallel():
    blind, mlp = _blind("runs_sigmoid/absdiff/seed0")
    lines = merge_lines(expand(blind, mlp).kinks, off=0.02)
    assert 8 <= len(lines) <= 64
    assert all(abs(abs(ln.normal @ np.array([1, -1]) / np.sqrt(2)) - 1) < 0.01 for ln in lines[:4])
    assert any(nearly_cancelled(ln) for ln in lines)


def test_merge_same_line_adds():
    k = [Kink(np.array([1., 0.]), -0.5, 2.0), Kink(np.array([1., 0.]), -0.51, -1.5)]
    (line,) = merge_lines(k, off=0.02)
    assert line.strength == pytest.approx(0.5) and line.gross == pytest.approx(3.5) and line.count == 2
    assert nearly_cancelled(line)                      # 0.5 < 0.25 * 3.5
    k2 = [Kink(np.array([1., 0.]), -0.5, 2.0), Kink(np.array([1., 0.]), -0.51, -0.2)]
    assert not nearly_cancelled(merge_lines(k2, off=0.02)[0])   # 1.8 >= 0.25 * 2.2
    far = [Kink(np.array([1., 0.]), -0.5, 2.0), Kink(np.array([1., 0.]), -0.6, 1.0)]
    assert len(merge_lines(far, off=0.02)) == 2 and len(merge_lines(far, off=0.1 + 1e-9)) == 1


# --- Task 2: refine, snap, write -------------------------------------------------------------------------------

def _tax_like(edge=193.3):
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 200)])
    x = np.linspace(0, 200, 2001)
    y = 0.1 * x + 0.1 * np.maximum(x - 20.2, 0) + 0.2 * np.maximum(x - 79.8, 0) - 0.3 * np.maximum(x - edge, 0)
    fit = RealFit(0.0, {"x0": 0.1}, [({"x0": 1.0}, 20.2, 0.1), ({"x0": 1.0}, 79.8, 0.2), ({"x0": 1.0}, edge, -0.3)])
    return spec, (x / 200)[:, None], y, fit


def _net(sizes, acts, seed=3):
    rng = np.random.default_rng(seed)
    return MLP([rng.normal(size=(o, i)).astype(np.float32) for i, o in zip(sizes, sizes[1:])],
               [rng.normal(size=o).astype(np.float32) for o in sizes[1:]], acts)


def _spec(n_in, output, n_out=1):
    return Spec(task="", output=output, n_outputs=n_out,
                inputs=[InputSpec(f"x{i}", f"x{i}", "float", -3, 3) for i in range(n_in)])


def test_kink_merge_tax_passes():
    blind, mlp = _blind("runs_sigmoid/tax/seed0")
    src, trace = kink_merge(blind, mlp)
    assert src is not None and check_program(src, mlp, blind).snap_pass
    assert trace["applies"] is None and trace["winner"]["m"] <= 8
    assert "relu" in src or "max" in src
    assert {p["off"] for p in trace["per_offset"]} == {0.02, 0.1}


def test_snap_real_coordinates_tax():
    spec, U, y, fit = _tax_like()
    snapped = snap_fit(fit, spec, 0.005, U, y)
    assert 192 < snapped.terms[-1][1] < 194
    assert {round(t, 6) for _, t, _ in snapped.terms[:2]} == {20.0, 80.0}


def test_snap_line_leaving_box_folds():
    spec, U, y, fit = _tax_like(edge=199.6)
    snapped = snap_fit(fit, spec, 0.005, U, y)  # (198.6, 200.6): nicest_in picks 200, the box edge
    assert len(snapped.terms) == 2
    src = fit_program(snapped, spec)
    assert src.count("relu(") + src.count("max(") <= 2


def test_to_real_matches_unit_fit():
    blind, mlp = _blind("runs_sigmoid/absdiff/seed0")
    lines = merge_lines(expand(blind, mlp).kinks, off=0.02)[:3]
    U = np.random.default_rng(2).uniform(0, 1, (3000, 2))
    lo, span = unit_box(blind)
    y = mlp.forward((lo + U * span).astype(np.float32))[:, 0].astype(np.float64)
    lines2, coefs, err = refine(lines, U, y)
    fit = to_real(lines2, coefs, blind)
    X = lo + U * span
    pred = fit.const + sum(c * X[:, i] for i, c in enumerate(fit.linear.values())) + sum(
        s * np.maximum(sum(n.get(f"x{i}", 0.0) * X[:, i] for i in range(2)) - t, 0) for n, t, s in fit.terms)
    assert np.abs(pred - y).max() == pytest.approx(err, rel=1e-6, abs=1e-9)
    assert all(max(abs(c) for c in n.values()) == pytest.approx(1.0) for n, _, _ in fit.terms)


def test_kink_merge_refuses():
    for sizes, acts, out in [([2, 4, 4, 1], ["sigmoid", "sigmoid"], "regression"),
                             ([2, 4, 3], ["sigmoid"], "argmax"),
                             ([2, 4, 1], ["relu"], "regression"),
                             ([2, 4, 1], ["tanh"], "regression")]:
        src, trace = kink_merge(_spec(2, out, n_out=sizes[-1]), _net(sizes, acts))
        assert src is None and trace["applies"]


# --- Task 3: the route in refactor -----------------------------------------------------------------------------

def test_refactor_trace_has_kink_merge():
    from nndc.ast_tool import refactor
    rd = Path("runs_sigmoid/tax/seed0")
    blind, mlp = _blind(rd)
    _, trace = refactor((rd / "dump.py").read_text(encoding="utf-8"), blind, mlp, kinks=True)
    assert "kink_merge" in trace and any(c["route"] == "kink-merge" for c in trace["candidates"])


def test_refactor_relu_has_no_kink_merge():
    from nndc.ast_tool import refactor
    from nndc.dump import dump_program
    mlp, spec = _net([2, 4, 1], ["relu"]), _spec(2, "regression")
    _, trace = refactor(dump_program(mlp, spec), spec, mlp)
    assert "kink_merge" not in trace and all(c["route"] != "kink-merge" for c in trace["candidates"])


def test_kink_merge_error_skips_only_the_route(monkeypatch):
    import nndc.ast_tool as ast_tool

    def boom(spec, mlp):
        raise ValueError("boom")
    monkeypatch.setattr(ast_tool, "kink_merge", boom)
    rd = Path("runs_sigmoid/tax/seed2")
    blind, mlp = _blind(rd)
    src, trace = ast_tool.refactor((rd / "dump.py").read_text(encoding="utf-8"), blind, mlp, kinks=True)
    assert src and all(c["route"] != "kink-merge" for c in trace["candidates"])
    assert any(ln.startswith("kink-merge: not applicable") and "boom" in ln for ln in trace["trace_lines"])


def test_snap_coefficients_relative():
    spec, U, y, fit = _tax_like()
    snapped = snap_fit(fit, spec, 0.05, U, y * 1.012)        # every slope 1.2% off a round value
    assert snapped.linear["x0"] == pytest.approx(0.1)
    assert [s for _, _, s in snapped.terms] == pytest.approx([0.1, 0.2, -0.3])


# --- ast3: kink merge v2 ---------------------------------------------------------------------------------------

def test_merge_antiparallel():
    k = [Kink(np.array([1., 0.]), -0.5, 2.0), Kink(np.array([-1., 0.]), 0.5, 1.0)]   # the same line, flipped
    (line,) = merge_lines(k, off=0.02)
    assert line.strength == pytest.approx(3.0) and line.count == 2
    assert line.normal @ np.array([1., 0.]) == pytest.approx(1.0) and line.offset == pytest.approx(-0.5)


def test_position_window_follows_line():
    spec = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 200),
                                                         InputSpec("x1", "x1", "float", 0, 1)])
    U = np.random.default_rng(3).uniform(0, 1, (3000, 2))
    X = U * np.array([200.0, 1.0])
    y = np.maximum(X[:, 1] - 0.433, 0)
    fit = RealFit(0.0, {"x0": 0.0, "x1": 0.0}, [({"x1": 1.0, "x0": 0.001}, 0.433, 1.0)])
    snapped = snap_fit(fit, spec, 0.05, U, y)
    assert len(snapped.terms) == 1 and 0.373 < snapped.terms[0][1] < 0.493   # not snapped to 0 by the x0 range


def test_tol0_folds():
    from nndc.heuristics.kinks import SNAP_LEVELS
    assert SNAP_LEVELS == (0.05, 0.02, 0.01, 0.005, 0.0)
    spec, U, y, fit = _tax_like(edge=205)                    # the third line lies outside the box
    snapped = snap_fit(fit, spec, 0.0, U, y)
    assert [t for _, t, _ in snapped.terms] == pytest.approx([20.2, 79.8])


def test_constant_rounded():
    spec, U, y, fit = _tax_like()
    snapped = snap_fit(fit, spec, 0.05, U, y + 0.0137)
    assert snapped.const == 0.0                               # nicest value within 0.05 * output range
    assert snap_fit(fit, spec, 0.0, U, y + 0.0137).const == pytest.approx(0.0137, abs=1e-6)
