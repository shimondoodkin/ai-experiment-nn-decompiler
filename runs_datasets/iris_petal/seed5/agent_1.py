def net(x0, x1):
    if x0 + x1 > 6.65:
        return 2
    if x0 + 1.75 * x1 < 3.85:
        return 0
    return 1
