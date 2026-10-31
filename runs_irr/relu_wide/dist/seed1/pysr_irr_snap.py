def net(x0, x1):
    return max(x0, abs(x1), ((0.8 * abs(x1)) + (0.6 * abs(x0))), ((0.35 * abs(x1)) + (0.95 * abs(x0))))
