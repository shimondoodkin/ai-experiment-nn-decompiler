def net(x0, x1):
    # Piecewise-linear bowl, roughly sqrt(1 + x0*x0 + x1*x1) (value ~0.92 at the origin, ~3 at the corners).
    # The function is even: net(-x0, -x1) == net(x0, x1); every hidden-unit pair became an abs().
    # cone at the origin: five folds through (0, 0)
    cone = (abs(0.29 * x0 + 0.03 * x1) + abs(0.26 * x0 - 0.07 * x1) + abs(0.12 * x0 - 0.2 * x1)
            + abs(0.06 * x0 - 0.25 * x1) + abs(0.18 * x0 + 0.26 * x1)) / 2
    # outer rings: folds that switch on away from the origin
    ring_x0 = relu(abs(0.25 * x0 - 0.02 * x1) - 0.21)
    ring_a = max(0.19, abs(0.09 * x0 - 0.22 * x1))
    ring_b = max(0.16, abs(0.14 * x0 + 0.2 * x1))
    return 0.575 + cone + ring_x0 + ring_a + ring_b
