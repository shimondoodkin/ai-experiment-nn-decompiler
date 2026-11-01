def net(x0, x1):
    return sqrt(max((1.0 + (x0 * x0) + (x1 * x1)), (1.1 + (x0 * (0.0 if (min(0.72, x0) < 0.0) else min(0.72, x0))))))
