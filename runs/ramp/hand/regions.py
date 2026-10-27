def net(x0):
    if x0 > 0:
        if x0 - 1.0 > 0:
            return 1.0
        else:
            return x0
    else:
        return 0.0
