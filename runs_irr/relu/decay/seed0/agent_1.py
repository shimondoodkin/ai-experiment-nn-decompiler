def net(x0):
    # Convex, decreasing piecewise-linear curve on [-1, 3] with kinks at x0 = 0, 1, 2.5.
    # Segment slopes, right to left: -0.2, -0.6, -1.2, -2.0 (each kink adds -0.4, -0.6, -0.8).
    tail = 1.33 - 0.2 * x0
    bend_far = 0.4 * relu(2.5 - x0)
    bend_mid = 0.6 * relu(1 - x0)
    bend_near = 0.8 * relu(-x0)
    return tail + bend_far + bend_mid + bend_near
