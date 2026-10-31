import numpy as np

import nndc.heuristics.prune as prune
from nndc.heuristics.prune import prune_candidate
from nndc.heuristics.sigmoid import minimax
from nndc.lang import evaluate, parse_program
from nndc.model import MLP
from nndc.spec import InputSpec, Spec

SPEC1 = Spec(task="", output="regression", inputs=[InputSpec("x0", "x0", "float", 0, 3)])


def net(units, act="sigmoid"):
    """A one-layer net from (out, w, b) per unit, one input."""
    a = np.array(units, dtype=np.float64)
    return MLP([a[:, 1:2].astype(np.float32), a[None, :, 0].astype(np.float32)],
               [a[:, 2].astype(np.float32), np.zeros(1, np.float32)], [act])


def dense_check(mlp, spec, frac=0.02):
    X = np.linspace(spec.inputs[0].lo, spec.inputs[0].hi, 20_001)[:, None]
    y = mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)
    tol = frac * float(np.ptp(y))

    def check(src):
        with np.errstate(all="ignore"):
            out = np.asarray(evaluate(parse_program(src, spec.arg_names), X), dtype=np.float64)
        err = np.where(np.isfinite(out), np.abs(out - y), np.inf)
        return bool(err.max() <= tol), X[err > tol][:5]
    return check


STRONG = [(2.0, 3.0, -3.0), (-1.5, -2.0, 1.0), (1.0, 4.0, -6.0)]


def test_prune_drops_redundant_units():
    mlp = net(STRONG + [(1e-4, 1.0, 0.0), (-1e-4, 2.0, -1.0), (1e-4, -3.0, 2.0)])
    src, info = prune_candidate(SPEC1, mlp, check=dense_check(mlp, SPEC1))
    assert src is not None and src.count("sigmoid(") <= 3, src
    assert dense_check(mlp, SPEC1)(src)[0]
    assert info["note"] is None and len(info["kept"]) <= 3


def test_prune_keeps_everything_when_needed():
    mlp = net([(2.0, 5.0, -5.0), (-2.0, 5.0, -10.0)])
    src, info = prune_candidate(SPEC1, mlp, check=dense_check(mlp, SPEC1))
    assert src is not None and src.count("sigmoid(") == 2, src
    assert info["kept"] == [0, 1]


def test_prune_lp_failure_returns_none(monkeypatch):
    monkeypatch.setattr(prune, "minimax", lambda A, y, method="highs": (np.zeros(A.shape[1]), float("inf")))
    mlp = net(STRONG)
    src, info = prune_candidate(SPEC1, mlp, check=dense_check(mlp, SPEC1))
    assert src is None and "LP" in info["note"]


def test_prune_numerical_failure_returns_none(monkeypatch):
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")
    monkeypatch.setattr(prune.np.linalg, "lstsq", broken)
    mlp = net(STRONG)
    src, info = prune_candidate(SPEC1, mlp, check=dense_check(mlp, SPEC1))
    assert src is None and "SVD" in info["note"]


def test_prune_not_for_binary_or_two_layers():
    mlp = net(STRONG)
    binary = Spec(task="", output="binary", inputs=SPEC1.inputs)
    assert prune_candidate(binary, mlp, check=lambda s: (True, np.empty((0, 1))))[0] is None
    two = MLP([np.ones((2, 1), np.float32), np.ones((2, 2), np.float32), np.ones((1, 2), np.float32)],
              [np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(1, np.float32)], ["sigmoid", "sigmoid"])
    assert prune_candidate(SPEC1, two, check=lambda s: (True, np.empty((0, 1))))[0] is None


def test_minimax_default_unchanged():
    rng = np.random.default_rng(3)
    A, y = rng.normal(size=(50, 3)), rng.normal(size=50)
    c1, e1 = minimax(A, y)
    c2, e2 = minimax(A, y, method="highs")
    assert e1 == e2 and np.array_equal(c1, c2)


SPEC2 = Spec(task="", output="regression",
             inputs=[InputSpec("x0", "x0", "float", -3, 3), InputSpec("x1", "x1", "float", -3, 3)])


def net2(units):
    """A one-layer sigmoid net from (out, w0, w1, b) per unit, two inputs."""
    a = np.array(units, dtype=np.float64)
    return MLP([a[:, 1:3].astype(np.float32), a[None, :, 0].astype(np.float32)],
               [a[:, 3].astype(np.float32), np.zeros(1, np.float32)], ["sigmoid"])


def grid_check(mlp, spec, frac=0.02):
    g = np.linspace(-3, 3, 201)
    X = np.stack([a.ravel() for a in np.meshgrid(g, g)], axis=1)
    y = mlp.forward(X.astype(np.float32))[:, 0].astype(np.float64)
    tol = frac * float(np.ptp(y))

    def check(src):
        out = np.asarray(evaluate(parse_program(src, spec.arg_names), X), dtype=np.float64)
        err = np.abs(out - y)
        return bool(err.max() <= tol), X[err > tol][:5]
    return check


def test_fold_shallow_sigmoid_units():
    shallow = [(1.0, 0.1, 0.0, 0.2), (1.0, 0.0, 0.1, -0.1), (1.0, 0.07, 0.07, 0.0), (1.0, -0.1, 0.0, 0.3)]
    mlp = net2(shallow + [(2.0, 3.0, 0.0, -1.0), (-2.0, 0.0, 3.0, 1.0)])
    src, info = prune_candidate(SPEC2, mlp, check=grid_check(mlp, SPEC2))
    assert info["folded"] == [0, 1, 2, 3], info
    assert src.count("sigmoid(") == 2 and grid_check(mlp, SPEC2)(src)[0], src
    assert None in info["atoms"] and len(info["atoms"]) < 6    # unneeded free terms are eliminated; the constant stays


def test_fold_relu_exact():
    mlp = net([(1.0, 1.0, 1.0), (2.0, 1.0, -1.5), (5.0, -1.0, -1.0)], act="relu")   # on, crossing, dead on [0, 3]
    src, info = prune_candidate(SPEC1, mlp, check=dense_check(mlp, SPEC1))
    assert info["folded"] == [0, 2] and src.count("relu(") == 1, (src, info)
    X = np.linspace(0, 3, 3001)[:, None]
    out = np.asarray(evaluate(parse_program(src, SPEC1.arg_names), X), dtype=np.float64)
    assert np.abs(out - mlp.forward(X.astype(np.float32))[:, 0]).max() < 1e-5
