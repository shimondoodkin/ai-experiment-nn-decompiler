def net(x0, x1):
    return max(0.4, x1, (0.0 if (((x0 * x0) + (x1 * x1)) < 0.0) else sqrt(((x0 * x0) + (x1 * x1)))))
