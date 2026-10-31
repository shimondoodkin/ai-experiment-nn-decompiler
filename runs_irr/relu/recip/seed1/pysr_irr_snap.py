def net(x0):
    return min((1.0 / x0), (1.25 * sqrt((1.0 / x0))), (sqrt(min(1.05, (1.5 / (x0)))) / (min(1.72, (x0 * x0)))))
