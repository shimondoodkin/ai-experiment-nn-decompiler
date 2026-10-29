def net(x0):
    # piecewise-linear: slope 0.19 below x0 = 80, slope 0.39 above (one hinge)
    base = 0.19 * x0 - 1
    extra = 0.2 * relu(x0 - 80)
    return base + extra
