def net(x0):
    return log(max(x0, (x0 * log(max((1.0 / sqrt(x0)), sqrt(x0))))))
