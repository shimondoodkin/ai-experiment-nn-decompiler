def net(x0):
    return min(sqrt(max(0.05, x0)), abs(max(x0, (2.0 * x0), (0.2 + x0 + (x0 * x0)))))
