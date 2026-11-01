def net(x0, x1):
    return max(abs(x0), abs(x1), ((0.875 * abs(x1)) + (0.55 * abs(x0))), ((0.875 * abs(x0)) + (0.45 * abs(x1))))
