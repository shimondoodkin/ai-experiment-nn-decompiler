def net(x0):
    # concave piecewise-linear approximation of log(x0) on [0.1, 5]: the lower envelope of 5 chords
    y = min(5.5 * x0 - 2.75, 2 * x0 - 1.75, 0.9 * (x0 - 1), 0.45 * x0 - 0.225, 0.25 * x0 + 0.375)
    return y
