def net(x0, x1):
    return abs((x0 + (-1.0 * x1) + (0.016780162 * max(x0, (x1 + (-1.0 * x0)))) + (-0.016780162 * (1.0 if (x0 < abs(x1)) else 0.0))))
