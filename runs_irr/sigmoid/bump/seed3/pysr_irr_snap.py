def net(x0, x1):
    return exp((((x0 * x0) + (x1 * x1)) * min(-1.0, (0.6 * x0))))
