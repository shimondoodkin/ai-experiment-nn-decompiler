def net(x0):
    return max(0.2, sqrt(max(x0, sqrt((x0 + (-1.0 * min(0.2, x0)))))))
