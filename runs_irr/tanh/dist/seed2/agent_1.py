# A bowl roughly like the distance from the origin, sqrt(x0^2 + x1^2), but with its own
# wobble of about +-0.2 (0.6 at the origin, about 4 at the corners), so it is kept as a
# small tanh net. The units come in pairs mirrored by (x0, x1) -> (-x1, -x0).
def net(x0, x1):
    d = x0 - x1
    side_a = tanh(0.6 * x0 - 0.1 * x1 + 1)
    side_b = tanh(0.1 * x0 - 0.6 * x1 + 1)
    diag_a = tanh(0.5 * x0 + 0.3 * x1 + 0.9)
    diag_b = tanh(0.3 * x0 + 0.5 * x1 - 0.9)
    anti = tanh(0.3 * d - 0.8)
    return 6.16 + 0.375 * d - 1.61 * (side_a + side_b) - 1.36 * (diag_a - diag_b) + 1.78 * anti
