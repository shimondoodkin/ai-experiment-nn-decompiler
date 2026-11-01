def net(x0):
    return (1.0 / max(x0, (0.75 * sqrt(x0)), (1.125 * sqrt(abs(x0)) * min(1.0, x0))))
