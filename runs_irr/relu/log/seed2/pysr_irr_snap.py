def net(x0):
    return max(-2.1, log(max(x0, log((2.75 * x0)), (max(2.75, log((x0 * x0))) * log(x0)))))
