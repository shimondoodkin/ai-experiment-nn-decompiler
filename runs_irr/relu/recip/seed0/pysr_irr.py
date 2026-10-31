def net(x0):
    return (1.0480912 * min(1.5955898, (sqrt((min(x0, exp((1.0 if (x0 > 1.5955898) else 0.0))) / (x0))) / (x0))))
