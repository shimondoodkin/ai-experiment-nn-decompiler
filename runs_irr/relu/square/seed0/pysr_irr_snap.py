def net(x0):
    return max(0.05, min((x0 * x0), (2.0 * abs(x0)), (-0.28 + (2.0 * (x0 * x0)))))
