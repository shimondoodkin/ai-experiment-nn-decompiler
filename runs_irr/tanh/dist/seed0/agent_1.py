def net(x0, x1):
    # Smoothed distance from the origin, roughly sqrt(x0*x0 + x1*x1),
    # with value ~0.48 at the origin and ~4.05 (not 4.24) in the corners.
    # Six tanh bumps, in near-mirror pairs (x0 -> -x0), plus a plane that
    # replaces the ten nearly-linear hidden units of the dump.
    steep_a = tanh(-1.1 * x0 - 0.5 * x1 - 0.9)
    steep_b = tanh(1.1 * x0 - 0.45 * x1 - 0.85)
    wide_a = tanh(-0.4 * x0 - 0.3 * x1 - 1.3)
    wide_b = tanh(0.4 * x0 - 0.25 * x1 - 1.25)
    up_a = tanh(-0.25 * x0 + 0.65 * x1 - 1.05)
    up_b = tanh(0.2 * x0 + 0.6 * x1 - 1.05)
    plane = 6.935 - 0.08 * x0 - 0.02 * x1
    return plane + 0.5 * (steep_a + steep_b) + 2.1 * wide_a + 2.5 * wide_b + up_a + 1.3 * up_b
