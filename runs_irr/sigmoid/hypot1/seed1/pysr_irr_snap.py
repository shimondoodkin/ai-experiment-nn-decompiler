def net(x0, x1):
    return sqrt(abs(((x0 * x0) + (-1.0 * max(-0.05, (-1.65 + abs(x1)))) + exp((0.0 if ((-0.32 + abs(x1)) < 0.0) else (-0.32 + abs(x1)))))))
