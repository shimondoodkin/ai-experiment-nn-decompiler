def net(x0):
    return min(exp(x0), (-0.4 + exp((x0 + max((0.8888888888888888 * x0), (min(exp(x0), (0.22 * exp((-1.0 * x0)))) * exp((-1.0 * x0))))))))
