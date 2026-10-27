def net(x0):
    # Convex piecewise-linear ramp: flat 0 below 3, then slopes 1/8, 1/5, 2/5
    # with kinks at x0 = 3, 21.5 and 80.
    return 0.125 * relu(x0 - 3) + 0.075 * relu(x0 - 21.5) + 0.2 * relu(x0 - 80)
