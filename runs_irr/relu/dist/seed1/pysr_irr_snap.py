def net(x0, x1):
    return sqrt(max((x0 * max(0.4, x0)), (abs((x1 * x1)) + max((x0 * x0), (-0.28 * x0 * x1)))))
