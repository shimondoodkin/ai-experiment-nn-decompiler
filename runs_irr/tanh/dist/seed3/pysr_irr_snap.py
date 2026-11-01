def net(x0, x1):
    return sqrt(((x1 * x1) + max(0.1, (0.0 if (((0.75 * (x0 * x0)) + max(0.125, (-0.95 + abs(x0)))) < 0.0) else ((0.75 * (x0 * x0)) + max(0.125, (-0.95 + abs(x0))))))))
