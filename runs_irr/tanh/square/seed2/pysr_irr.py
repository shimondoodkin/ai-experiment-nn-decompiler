def net(x0):
    return abs((x0 * (min(1.9389527, abs(x0)) + (-0.027511438 if ((x0 * x0) < 1.4849948) else 0.0))))
