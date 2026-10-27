def net(x0, x1, x2):
    return max(min(x1, x2), min(x0, max(x1, x2)))
