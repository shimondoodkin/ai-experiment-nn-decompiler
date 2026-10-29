"""Kink merge: sigmoids expanded into ReLU kinks, kinks merged by line (nndc/heuristics/kinks.py)."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from experiments.tasks import uniform_sample
from nndc.heuristics.kinks import (Kink, evaluate_expansion, expand, merge_lines, nearly_cancelled, unit_box)
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
