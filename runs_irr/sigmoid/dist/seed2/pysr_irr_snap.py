def net(x0, x1):
    return max(x1, sqrt(max(0.2, ((x0 * max(-2.875, x0)) + (x1 * max(-2.875, x1))))))
