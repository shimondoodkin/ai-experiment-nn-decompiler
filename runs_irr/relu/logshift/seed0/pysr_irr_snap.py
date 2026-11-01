def net(x0):
    return log(max((2.0 + x0), (0.6 + (1.4 * x0)), log((2.0 + x0)), min(3.125, (1.3 + (1.8 * x0)))))
