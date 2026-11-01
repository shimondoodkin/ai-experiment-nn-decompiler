def net(x0, x1):
    return sqrt((abs((x0 * x0)) + (0.0 if ((max(-0.32, (-0.55 * abs(x1))) + exp((0.8 * abs(x1)))) < 0.0) else (max(-0.32, (-0.55 * abs(x1))) + exp((0.8 * abs(x1)))))))
