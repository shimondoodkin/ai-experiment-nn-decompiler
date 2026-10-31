def net(x0, x1):
    return max(0.0, exp(((-1.0 * (x1 * x1)) + (-1.0 * x0 * (x0 + (-0.01 * abs((x0 * max(-1.42, x0)))))))))
