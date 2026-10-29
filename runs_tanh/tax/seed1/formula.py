def net(x0):
    return 0.1 * min(x0, 20) + 0.2 * min(max(x0 - 20, 0), 60) + 0.4 * max(x0 - 80, 0)
