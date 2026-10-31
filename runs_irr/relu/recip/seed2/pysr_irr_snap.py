def net(x0):
    return (max(0.95, (1.0 if (x0 < 1.7) else 0.0), (1.0 if (x0 > 2.0) else 0.0)) / (max(0.55, x0, (1.0 if (x0 > 0.875) else 0.0))))
