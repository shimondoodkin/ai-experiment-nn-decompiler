def net(x0, x1):
    return max(0.0, (1.1 + (0.55 * min(-0.28, x1, (-1.0 * x1))) + (-0.55 * abs(x0))))
