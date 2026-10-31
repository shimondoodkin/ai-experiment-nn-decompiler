def net(x0, x1):
    return (0.08456718 + max(abs(x0), abs(x1), (1.4142135623730951 * sqrt(abs((x0 * x1))))))
