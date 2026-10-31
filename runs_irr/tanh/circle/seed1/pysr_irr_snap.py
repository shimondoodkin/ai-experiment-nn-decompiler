def net(x0, x1):
    return ((-1.0 * exp(abs(x0))) + (-1.0 * exp(abs(x1))) + max(log((1.0 + abs(x1))), (26.0 + (-1.4 * exp((abs(x0) + abs(x1)))))))
