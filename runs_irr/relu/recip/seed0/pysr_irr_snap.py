def net(x0):
    return (sqrt(((0.0 if (min(1.6, (1.1 / (x0))) < 0.0) else min(1.6, (1.1 / (x0)))) * exp((min(1.08, exp((1.0 if (x0 > 1.6) else 0.0))) * min(log(x0), (1.0 if (x0 > 1.55) else 0.0)))))) / (x0))
