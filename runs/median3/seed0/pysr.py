def net(x0, x1, x2):
    return min(max(x0, x2), max(x1, min(x0, x2)))
