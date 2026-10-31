def net(x0, x1):
    return (0.05 + max(abs(x0), abs(x1), ((0.7 * abs(x0)) + (0.7 * abs(x1)))))
