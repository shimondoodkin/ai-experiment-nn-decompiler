def net(x0, x1):
    return (min(exp((-1.0 * (x0 * x0))), exp((x0 * ((x0 * x0) + (0.9 * x0))))) * exp((-1.0 * (x1 * x1))))
