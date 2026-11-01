def net(x0, x1):
    return sqrt(((x1 * x1) + (x1 * max(0.0, min(x0, (-0.012062449 * x0)))) + (0.0 if ((x0 * x0) < -1.0) else (1.0 + (x0 * x0)))))
