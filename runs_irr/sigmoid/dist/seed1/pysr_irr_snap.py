def net(x0, x1):
    return abs(sqrt(max((0.2 + abs((x0 * x1))), ((x0 * x0) + (x1 * x1)))))
