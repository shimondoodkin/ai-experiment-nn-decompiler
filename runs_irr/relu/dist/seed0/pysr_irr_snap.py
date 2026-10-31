def net(x0, x1):
    return max(abs(x0), abs(x1), (0.875 * max(((0.6 * abs(x0)) + abs(x1)), ((0.6 * abs(x1)) + abs(x0)))))
