def net(x0):
    return min((-1.05 + x0), log(max(0.125, x0, (1.25 + log(x0)), (sqrt(x0) + min(0.875, log(x0))))))
