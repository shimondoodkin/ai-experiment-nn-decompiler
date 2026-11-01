def net(x0):
    # The net is a 14-kink piecewise-linear approximation of sqrt(x0) on [0, 4]
    # (hits 1, 1.414, 1.732, 2 at x0 = 1, 2, 3, 4). Near 0 a ReLU net cannot follow
    # sqrt's vertical slope, so it starts at about 0.056 instead of 0; the floor
    # keeps the program within tolerance there.
    root = sqrt(x0)
    return max(root, 0.05)
