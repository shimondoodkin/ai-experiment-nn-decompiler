import numpy as np

from nndc.heuristics.conic import _clean, conic_candidate
from nndc.lang import evaluate, parse_program
from nndc.spec import InputSpec, Spec


def spec_2d(lo, hi, output="binary"):
    return Spec(task="", output=output,
                inputs=[InputSpec("x0", "x0", "float", lo, hi), InputSpec("x1", "x1", "float", lo, hi)])


def decision_check(decide, spec, frac=0.002):
    """A stand-in for the checker: at most frac of a dense sample decided differently."""
    rng = np.random.default_rng(1)
    X = np.stack([rng.uniform(i.lo, i.hi, 50_000) for i in spec.inputs], axis=1)
    want = decide(X)

    def check(src: str) -> bool:
        got = (evaluate(parse_program(src, spec.arg_names), X) > spec.threshold).astype(int)
        return bool((got != want).mean() <= frac)
    return check


def run(decide, spec):
    return conic_candidate(spec, None, decide=decide, check=decision_check(decide, spec))


def test_circle_snaps():
    src, info = run(lambda X: (X[:, 0] ** 2 + X[:, 1] ** 2 < 4).astype(int), spec_2d(-3, 3))
    assert info["passed"] and src == "def net(x0, x1):\n    return x0 * x0 + x1 * x1 < 4\n", src


def test_offcentre_ellipse_passes():
    src, info = run(lambda X: ((X[:, 0] - 0.5) ** 2 / 4 + X[:, 1] ** 2 < 1).astype(int), spec_2d(-3, 3))
    assert info["passed"] and info["n_boundary"] >= 20, info


def test_side_is_taken_from_the_net():
    src, info = run(lambda X: (X[:, 0] ** 2 + X[:, 1] ** 2 > 4).astype(int), spec_2d(-3, 3))
    assert info["passed"] and src == "def net(x0, x1):\n    return x0 * x0 + x1 * x1 > 4\n", src


def test_no_boundary_returns_none():
    src, info = run(lambda X: np.ones(len(X), int), spec_2d(-3, 3))
    assert src is None and not info["passed"]


def test_not_applicable_to_regression_or_one_input():
    one = Spec(task="", output="binary", inputs=[InputSpec("x0", "x0", "float", 0, 1)])
    assert conic_candidate(spec_2d(0, 1, "regression"), None, decide=lambda X: X[:, 0] > 0, check=lambda s: True)[0] is None
    assert conic_candidate(one, None, decide=lambda X: X[:, 0] > 0.5, check=lambda s: True)[0] is None


CLEAN = "def net(x0, x1):\n    return x0 * x0 + x1 * x1 < 4\n"


def dirty_circle(X):
    return (X[:, 0] ** 2 + X[:, 1] ** 2 - 0.01 * X[:, 0] + 0.01 * X[:, 1] < 3.977).astype(int)


def test_clean_conic_drops_small_terms():
    spec = spec_2d(-3, 3)
    src, info = conic_candidate(spec, None, decide=dirty_circle, check=decision_check(dirty_circle, spec, 0.01))
    assert src == CLEAN, (src, info)
    assert info["clean"]["passed"] and info["passed"]


def test_clean_conic_keeps_direction_and_sign():
    def outside(X):
        return (X[:, 0] ** 2 + X[:, 1] ** 2 - 0.01 * X[:, 0] > 3.977).astype(int)
    spec = spec_2d(-3, 3)
    src, _ = conic_candidate(spec, None, decide=outside, check=decision_check(outside, spec, 0.01))
    assert src == "def net(x0, x1):\n    return x0 * x0 + x1 * x1 > 4\n", src
    t = np.linspace(0, 2 * np.pi, 400)                     # boundary of x0² - x1² = -1.02 (a hyperbola), negative c
    P = np.stack([np.sqrt(1.02) * np.sinh(t / 3), np.sqrt(1.02) * np.cosh(t / 3)], axis=1)
    c = _clean(np.array([1.0, -1.0, 0.0, 0.005, 0.0, -1.02]), P)
    assert c[3] == 0 and c[5] == -1


def test_clean_conic_falls_back():
    src, info = conic_candidate(spec_2d(-3, 3), None, decide=dirty_circle, check=lambda s: s != CLEAN)
    assert src != CLEAN and info["passed"]
    assert info["clean"] == {"src": CLEAN, "passed": False}
