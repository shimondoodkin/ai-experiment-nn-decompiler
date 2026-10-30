import numpy as np

from nndc.heuristics.families import MAX_TERMS, family_candidates
from nndc.lang import evaluate, parse_program
from nndc.spec import InputSpec, Spec


def spec_1d(lo, hi, output="regression"):
    return Spec(task="", output=output, inputs=[InputSpec("x0", "x0", "float", lo, hi)])


def spec_2d(lo, hi, output="regression"):
    return Spec(task="", output=output,
                inputs=[InputSpec("x0", "x0", "float", lo, hi), InputSpec("x1", "x1", "float", lo, hi)])


def max_err_check(f, spec, frac=0.02):
    """A stand-in for the checker: max error on a dense sample within frac of f's range."""
    rng = np.random.default_rng(1)
    X = np.stack([rng.uniform(i.lo, i.hi, 50_000) for i in spec.inputs], axis=1)
    y = f(X)
    tol = frac * float(y.max() - y.min())

    def check(src: str) -> bool:
        with np.errstate(all="ignore"):
            out = evaluate(parse_program(src, spec.arg_names), X)
        return bool(np.isfinite(out).all() and np.abs(out - y).max() <= tol)
    return check


def run(f, spec):
    return family_candidates(spec, None, predict=f, check=max_err_check(f, spec))


def test_exp_net_snaps_to_exp():
    progs, info = run(lambda X: np.exp(X[:, 0]), spec_1d(-2, 2))
    assert ("exp", "def net(x0):\n    return exp(x0)\n") in progs
    assert {r["family"] for r in info} >= {"identity", "square", "exp", "log", "sqrt", "reciprocal"}


def test_shifted_log_snaps_with_offset():
    progs, _ = run(lambda X: np.log(X[:, 0] + 2), spec_1d(-1.5, 4))
    assert any(f == "log" and "log(x0 + 2)" in s for f, s in progs), progs


def test_correction_term_added_when_flattened():
    progs, info = run(lambda X: np.sqrt(X[:, 0]) - 0.15 * np.maximum(0, X[:, 0] - 3), spec_1d(0, 4))
    row = next(r for r in info if r["family"] == "sqrt")
    assert row["passed"] and 1 <= row["terms"] <= MAX_TERMS and "relu(" in row["src"], row


def test_2d_radial_exp_uses_general_quadratic():
    progs, _ = run(lambda X: np.exp(-((X[:, 0] - 0.5) ** 2 + X[:, 1] ** 2)), spec_2d(-3, 3))
    assert any(fam == "exp" for fam, _ in progs), progs


def test_2d_product_is_quadratic():
    progs, _ = run(lambda X: X[:, 0] * X[:, 1], spec_2d(-2, 2))
    assert ("quadratic", "def net(x0, x1):\n    return x0 * x1\n") in progs


def test_nonfinite_family_is_dropped_not_raised():
    _, info = run(lambda X: X[:, 0] ** 3, spec_1d(-2, 2))
    assert info and all(np.isfinite(r["fit_err"]) or not r["passed"] for r in info)


def test_constant_net_returns_nothing():
    assert run(lambda X: np.full(len(X), 0.3), spec_1d(0, 1)) == ([], [])


def test_lawson_lowers_max_error():
    # sqrt(x + 0.1) plus a bump near x = 3: least squares spreads the error, minimax lowers the worst point
    f = lambda X: np.sqrt(X[:, 0] + 0.1) + 0.06 * np.exp(-((X[:, 0] - 3) / 0.3) ** 2)   # noqa: E731
    _, info = run(f, spec_1d(0, 4))
    row = next(r for r in info if r["family"] == "sqrt")
    assert row["minimax_err"] < row["fit_err"]          # fit_err stays the least-squares error


def test_lawson_on_exact_fit_stops():
    _, info = run(lambda X: np.exp(X[:, 0]), spec_1d(-2, 2))
    assert next(r for r in info if r["family"] == "exp")["minimax_err"] <= 1e-9


def test_lawson_keeps_ls_fit_when_minimax_goes_nonfinite(monkeypatch):
    import nndc.heuristics.families as fam
    real = fam._refit

    def refit(m, X, y, R, nfev, weights=None):
        v = real(m, X, y, R, nfev)
        return np.full_like(v, np.nan) if weights is not None else v
    monkeypatch.setattr(fam, "_refit", refit)
    progs, info = run(lambda X: np.exp(X[:, 0]), spec_1d(-2, 2))
    row = next(r for r in info if r["family"] == "exp")
    assert row["minimax_err"] == row["fit_err"] and row["passed"]


def test_not_applicable_to_binary_or_three_inputs():
    three = Spec(task="", output="regression", inputs=[InputSpec(f"x{k}", f"x{k}", "float", 0, 1) for k in range(3)])
    assert family_candidates(spec_1d(0, 1, "binary"), None, predict=lambda X: X[:, 0], check=lambda s: True) == ([], [])
    assert family_candidates(three, None, predict=lambda X: X[:, 0], check=lambda s: True) == ([], [])
