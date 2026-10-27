def net(x0):
    # convex piecewise-linear curve: flat-ish start, slope ~0.185 from x0=2, slope 0.4 from x0~78
    return max(-0.5 - x0 / 8, 0.185 * x0 - 1.12, 0.4 * x0 - 18)
