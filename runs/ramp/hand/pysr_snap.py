def net(x0):
    return (0.0 if (min(1.0, x0) < 0.0) else min(1.0, x0))
