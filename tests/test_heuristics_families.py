import numpy as np

from nndc.heuristics.families import (MARGIN_RETRIES, MAX_TERMS, N_BORDER, N_EXTREME, N_FIT_2D, _fit_sample,
                                     faithful_fit, family_candidates, near_formula)
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

    def check(src: str) -> tuple[bool, np.ndarray]:
        with np.errstate(all="ignore"):
            out = evaluate(parse_program(src, spec.arg_names), X)
        err = np.where(np.isfinite(out), np.abs(out - y), np.inf)
        bad = err > tol
        return bool(not bad.any()), X[np.argsort(-err)[:5]][err[np.argsort(-err)[:5]] > tol]
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
    assert family_candidates(spec_1d(0, 1, "binary"), None, predict=lambda X: X[:, 0], check=lambda s: (True, np.empty((0, 1)))) == ([], [])
    assert family_candidates(three, None, predict=lambda X: X[:, 0], check=lambda s: (True, np.empty((0, 1)))) == ([], [])


def test_near_formula_prefers_fewer_terms_within_3pct():
    rows = [{"family": "reciprocal", "near": [{"terms": 4, "src": "r4", "err": 0.021, "nodes": 30}]},
            {"family": "sqrt", "near": [{"terms": 0, "src": "s0", "err": 0.025, "nodes": 9},
                                       {"terms": 4, "src": "s4", "err": 0.022, "nodes": 31}]}]
    assert near_formula(rows, 2)["src"] == "s0"


def test_near_formula_falls_back_to_smallest_error():
    rows = [{"family": "exp", "near": [{"terms": 0, "src": "e0", "err": 0.06, "nodes": 6}]},
            {"family": "log", "near": [{"terms": 2, "src": "l2", "err": 0.04, "nodes": 15}]}]
    assert near_formula(rows, 1)["src"] == "l2"


def test_identity_is_never_near_in_1d_but_quadratic_is_in_2d():
    assert near_formula([{"family": "identity", "near": [{"terms": 0, "src": "i", "err": 0.01, "nodes": 3}]}], 1) is None
    q = near_formula([{"family": "quadratic", "near": [{"terms": 0, "src": "q", "err": 0.025, "nodes": 5}]}], 2)
    assert q == {"family": "quadratic", "src": "q", "err": 0.025, "terms": 0, "nodes": 5}


def test_near_formula_skips_rows_without_near():
    assert near_formula([{"family": "conic", "passed": False}, {"family": "exp", "near": []}], 2) is None


def test_near_rows_filled_when_family_fails():
    f = lambda X: np.sqrt(X[:, 0]) + 0.05 * np.sin(6 * X[:, 0])     # noqa: E731  sqrt plus ripple: fails at 2%
    progs, info = run(f, spec_1d(0, 4))
    assert not any(fam == "sqrt" for fam, _ in progs)
    near = next(r for r in info if r["family"] == "sqrt")["near"]
    assert near and all(0 < c["err"] for c in near) and [c["terms"] for c in near] == sorted({c["terms"] for c in near})
    assert all(c["src"].startswith("def net(x0):") and c["nodes"] > 0 for c in near)


def test_near_rows_empty_for_a_passing_family():
    _, info = run(lambda X: np.exp(X[:, 0]), spec_1d(-2, 2))
    assert next(r for r in info if r["family"] == "exp")["near"] == []


def test_faithful_fit_keeps_the_dirt():
    ff = faithful_fit(spec_1d(0, 2), lambda X: 1.03 * np.exp(0.98 * X[:, 0]) + 0.02, "exp")
    assert "exp(" in ff["src"] and ff["src"] != ff["clean_src"]
    assert ff["err"] < 1e-3                       # the faithful fit follows the net
    assert 0 < ff["distortion"] <= 0.021          # the clean formula is within the tolerance of it
    effects = [d["effect"] for d in ff["dirt"]]
    assert effects and effects == sorted(effects, reverse=True) and all(e > 0 for e in effects)
    assert {"term", "clean", "faithful", "effect"} <= set(ff["dirt"][0])


def test_faithful_fit_labels_2d_terms():
    ff = faithful_fit(spec_2d(-2, 2), lambda X: 1.02 * X[:, 0] ** 2 + X[:, 1] ** 2 + 0.05 * X[:, 0] * X[:, 1],
                      "quadratic")
    assert ff["err"] < 1e-3
    assert {d["term"] for d in ff["dirt"]} & {"x0²", "x0·x1"}


def test_faithful_fit_not_a_family():
    assert faithful_fit(spec_1d(0, 2), lambda X: X[:, 0], "conic") is None


def test_fit_sample_has_border_and_corners():
    X, y = _fit_sample(spec_2d(-3, 3), lambda X: X[:, 0] + X[:, 1])
    assert len(X) == N_FIT_2D + 4 * N_BORDER + N_EXTREME == len(y)
    for c in ([-3, -3], [-3, 3], [3, -3], [3, 3]):
        assert (np.abs(X - c).max(axis=1) < 1e-12).any(), c


def exp_stub(results):
    """A check that answers the exp program from results in turn (then passes) and fails every other program
    without counterexamples."""
    calls = iter(results)

    def check(src):
        if "exp(" not in src:
            return False, np.empty((0, 1))
        return next(calls, (True, np.empty((0, 1))))
    return check


def exp_row(check):
    _, info = family_candidates(spec_1d(-2, 2), None, predict=lambda X: np.exp(X[:, 0]), check=check)
    return next(r for r in info if r["family"] == "exp")


def test_margin_retry_after_checker_failure():
    r = exp_row(exp_stub([(False, np.array([[2.0]]))]))
    assert r["retries"] == 1 and r["margin"] == 0.8 and r["passed"]


def test_no_retry_without_counterexamples():
    r = exp_row(exp_stub([(False, np.empty((0, 1)))]))
    assert r["retries"] == 0 and r["margin"] == 1.0


def test_at_most_two_retries():
    r = exp_row(exp_stub([(False, np.array([[2.0]]))] * 50))
    assert r["retries"] == MARGIN_RETRIES == 2 and r["margin"] == 0.6
