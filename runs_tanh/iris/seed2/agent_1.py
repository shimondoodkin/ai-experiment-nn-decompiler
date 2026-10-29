# Refactored from dump.py: 4 inputs, 3 classes.  The input ranges match the Iris measurements
# (x0, x1 sepal-like; x2, x3 petal-like), so the units are named after what they compare.
# Scores are relative to class 1 (class 1 has score 0).
def net(x0, x1, x2, x3):
    # main gate: x2 + x3 large relative to x1 -> not class 0 (merge of 3 near-duplicate units)
    petal_vs_x1 = tanh(-0.03 * x0 - 1.03 * x1 + 1.12 * x2 + 1.2 * x3 - 0.6)
    # class-2 detectors: x2, x3 large relative to x0 / x1
    petal_vs_x0 = tanh(-0.65 * x0 + 1.02 * x2 + 1.5 * x3 - 2.5)
    x3_vs_x1 = tanh(0.1 * x0 - 0.9 * x1 + 0.5 * x2 + 1.25 * x3 - 2)
    # class-1 detector: x2, x3 small
    small_petal = tanh(0.22 * x0 + 0.7 * x1 - 0.77 * x2 - 1.4 * x3 + 2.6)
    score0 = 1.5 - 11 * petal_vs_x1 + 2 * petal_vs_x0 - 1.5 * small_petal
    score2 = 5 * petal_vs_x0 + 3 * x3_vs_x1 - 5 * petal_vs_x1 - 6 * small_petal
    return argmax(score0, 0, score2)
