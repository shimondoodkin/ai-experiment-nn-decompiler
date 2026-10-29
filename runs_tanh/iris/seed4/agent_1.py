def net(x0, x1, x2, x3):
    # class 0: x3 below a plane
    if x3 + 5 / 6 * x2 < 0.125 * x0 + 0.625 * x1 + 0.37:
        return 0
    # class 2 vs 1: x3 above a plane bent by a parabola in u
    u = x1 - 0.65 * x0 + 0.15 * x2
    if x3 > 2.1 + 0.18 * x0 + 0.43 * x1 - 0.56 * x2 + u * u / 24:
        return 2
    return 1
