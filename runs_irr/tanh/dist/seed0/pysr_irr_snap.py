def net(x0, x1):
    return sqrt(max(0.3, x0, abs(x0), (-0.125 + (x0 * x0) + max((x1 * x1), (-0.125 + (-1.0 * x1))))))
