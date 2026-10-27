def net(x0, x1):
    return (x0 + x1 + (0.0 if ((x0 + (-1.0 * x1)) < 0.0) else (x0 + (-1.0 * x1))))
