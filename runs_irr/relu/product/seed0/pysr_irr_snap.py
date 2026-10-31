def net(x0, x1):
    return (x1 * (x0 + (-0.00037924908 * (x1 * x1) * exp((x0 * (0.0 if (x0 < 0.0) else x0))))))
