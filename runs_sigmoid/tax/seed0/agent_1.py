def net(x0):
    # Convex piecewise-linear ramp: slope 0.18 below the knee, slope 0.39 above it.
    # The two lines cross at x0 = 75 (0.18*75 - 0.7 = 0.39*75 - 16.45 = 12.8).
    low_ramp = 0.18 * x0 - 0.7
    high_ramp = 0.39 * x0 - 16.45
    return max(low_ramp, high_ramp)
