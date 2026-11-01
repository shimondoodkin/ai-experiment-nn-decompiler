def net(x0, x1):
    return max(abs(x0), abs(x1), ((0.5609089 * abs(x0)) + (0.8448855 * abs(x1))), ((0.4360266 * abs(x1)) + (0.9107827 * abs(x0))))
