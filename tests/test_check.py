import numpy as np

from nndc.check import build_input_set, check_program, format_result
from nndc.model import MLP
from nndc.spec import InputSpec, Spec
from tests.helpers import hand


def absdiff_net():   # relu(a-b) + relu(b-a)
    return MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[1, 1]], np.float32)],
               [np.zeros(2, np.float32), np.zeros(1, np.float32)])


REG = Spec(task="t", output="regression",
           inputs=[InputSpec("x0", "a", "float", -3, 3), InputSpec("x1", "b", "float", -3, 3)])


def test_exact_formula_passes_exact():
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1)", absdiff_net(), REG)
    assert r.exact_pass and r.snap_pass and r.label == "empirical" and r.error is None


def test_slightly_off_passes_snap_not_exact():
    r = check_program("def net(x0, x1):\n    return 1.01 * abs(x0 - x1)", absdiff_net(), REG)
    assert not r.exact_pass and r.snap_pass      # max err 0.06 <= 0.02*6 = 0.12


def test_wrong_fails_with_counterexamples():
    r = check_program("def net(x0, x1):\n    return x0 - x1", absdiff_net(), REG)
    assert not r.snap_pass and len(r.counterexamples) > 0
    assert set(r.counterexamples[0]["inputs"]) == {"x0", "x1"}


def test_program_error_is_reported_not_raised():
    r = check_program("import os\ndef net(x0, x1):\n    return 0", absdiff_net(), REG)
    assert r.error and "Import" in r.error and not r.snap_pass
    assert format_result(r, REG).startswith("ERROR")


def test_nan_counts_as_disagreement():
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1) * (1 if x0 < 2.9 else 1e308) * 10",
                      absdiff_net(), REG)
    assert not r.exact_pass


def test_binary_snap_zone():
    # net: 0.2 - |a-b|  (class 1 iff |a-b| < 0.2); program uses 0.21: disagrees only near the boundary
    net = MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[-1, -1]], np.float32)],
              [np.zeros(2, np.float32), np.array([0.2], np.float32)])
    spec = Spec(task="t", output="binary",
                inputs=[InputSpec("x0", "a", "float", -2, 2), InputSpec("x1", "b", "float", -2, 2)])
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.21", net, spec)
    assert not r.exact_pass and r.snap_pass and r.disagreements_inside > 0 and r.disagreements_outside == 0
    r2 = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.6", net, spec)
    assert not r2.snap_pass and r2.disagreements_outside > 0


def test_float_ties_on_the_decision_line_do_not_break_exact():
    # the net computes in float32, programs in float64: on the decision line itself the class is a
    # rounding tie, which exact mode tolerates like regression's atol/rtol
    net = MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[-1, -1]], np.float32)],
              [np.zeros(2, np.float32), np.array([0.2], np.float32)])
    spec = Spec(task="t", output="binary",
                inputs=[InputSpec("x0", "a", "float", -2, 2), InputSpec("x1", "b", "float", -2, 2)])
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.2", net, spec)
    assert r.exact_pass


def test_float_ties_scale_with_large_logits():
    # trained classifiers have large logits; float32 rounding near the line grows with them
    net = MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[-1000, -1000]], np.float32)],
              [np.zeros(2, np.float32), np.array([200], np.float32)])
    spec = Spec(task="t", output="binary",
                inputs=[InputSpec("x0", "a", "float", -2, 2), InputSpec("x1", "b", "float", -2, 2)])
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.2", net, spec)
    assert r.exact_pass


def _large_logit_neareq():
    net = MLP([np.array([[1, -1], [-1, 1]], np.float32), np.array([[-1000, -1000]], np.float32)],
              [np.zeros(2, np.float32), np.array([200], np.float32)])
    spec = Spec(task="t", output="binary",
                inputs=[InputSpec("x0", "a", "float", -2, 2), InputSpec("x1", "b", "float", -2, 2)])
    return net, spec


def test_shifted_threshold_is_not_exact():
    # a 1e-4 threshold shift is a real disagreement, far above float32 rounding of these logits
    net, spec = _large_logit_neareq()
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.2001", net, spec)
    assert not r.exact_pass and r.snap_pass


def test_zone_fraction_is_a_fraction_of_the_domain():
    net, spec = _large_logit_neareq()
    r = check_program("def net(x0, x1):\n    return abs(x0 - x1) < 0.2", net, spec)
    X = np.random.default_rng(7).uniform(-2, 2, (400_000, 2))
    expected = (np.abs(200 - 1000 * np.abs(X[:, 0] - X[:, 1])) < r.margin).mean()
    assert abs(r.zone_fraction - expected) < 0.005


def test_exhaustive_label_for_small_discrete():
    xor = MLP([np.array([[1, 1], [1, 1]], np.float32), np.array([[1, -2]], np.float32)],
              [np.array([0, -1], np.float32), np.array([-0.5], np.float32)])
    spec = Spec(task="t", output="binary",
                inputs=[InputSpec("x0", "p", "bool", 0, 1), InputSpec("x1", "q", "bool", 0, 1)])
    r = check_program("def net(x0, x1):\n    return x0 != x1", xor, spec)
    assert r.label == "exhaustive" and r.n_inputs == 4 and r.exact_pass


def test_argmax_program():
    net = MLP([np.array([[1, 0], [0, 1]], np.float32), np.array([[1, 0], [0, 1]], np.float32)],
              [np.zeros(2, np.float32), np.zeros(2, np.float32)])     # outputs relu(x0), relu(x1)
    spec = Spec(task="t", output="argmax", n_outputs=2,
                inputs=[InputSpec("x0", "a", "float", 0, 1), InputSpec("x1", "b", "float", 0, 1)])
    assert check_program("def net(x0, x1):\n    return 1 if x1 > x0 else 0", net, spec).exact_pass
    r = check_program("def net(x0, x1):\n    return 0", net, spec)
    assert not r.snap_pass and r.disagreements_outside > 0


def test_rare_branch_missed_by_random_but_caught_by_boundary_points():
    # bump of height 1 and width 4e-6 at x=1 on [-5, 5], exactly 0 elsewhere (no float cancellation):
    # h = relu(x-1), relu(1-x); g = relu(e - h0 - h1); out = g / e
    e = 2e-6
    net = MLP([np.array([[1], [-1]], np.float32), np.array([[-1, -1]], np.float32),
               np.array([[1 / e]], np.float32)],
              [np.array([-1, 1], np.float32), np.array([e], np.float32), np.zeros(1, np.float32)])
    spec = Spec(task="t", output="regression", inputs=[InputSpec("x0", "x", "float", -5, 5)])
    Xr = np.random.default_rng(0).uniform(-5, 5, (200_000, 1))
    assert (net.forward(Xr.astype(np.float32))[:, 0] > 0.1).sum() == 0      # pure random sampling misses it
    r = check_program("def net(x0):\n    return 0", net, spec)
    assert not r.snap_pass                                                   # boundary points catch it


def test_extra_inputs_are_replayed():
    X, _ = build_input_set(absdiff_net(), REG, extra=np.array([[2.5, -2.5]]))
    assert any((row == [2.5, -2.5]).all() for row in X)


def test_constant_regression_net_passes_snap_when_exact():
    # every unit dead: output is a constant, so the snap tolerance (2% of a zero range) is 0
    net = MLP([np.array([[1.0]], np.float32), np.array([[3.0]], np.float32)],
              [np.array([-5.0], np.float32), np.array([0.7], np.float32)])
    spec = Spec(task="t", output="regression", inputs=[InputSpec("x0", "x", "float", 0, 1)])
    r = check_program("def net(x0):\n    return 0.7", net, spec)
    assert r.exact_pass and r.snap_pass


def test_no_boundary_points_option():
    mlp, spec = hand("window")
    full, _ = build_input_set(mlp, spec)
    plain, _ = build_input_set(mlp, spec, boundary=False)
    assert plain.shape[0] < full.shape[0]
    r = check_program("def net(x0):\n    return abs(x0) < 0.5\n", mlp, spec, boundary=False)
    assert r.snap_pass and r.n_inputs == plain.shape[0]
