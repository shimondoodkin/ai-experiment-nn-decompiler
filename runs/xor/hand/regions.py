def net(x0, x1):
    if x0 + x1 - 1.0 > 0:
        return -x0 - x1 + 1.5
    else:
        return x0 + x1 - 0.5
