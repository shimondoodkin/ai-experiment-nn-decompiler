def net(x0, x1):
    if x1 <= 0.5:
        if x0 <= 0.5:
            return 0
        else:
            return 1
    else:
        if x0 <= 0.5:
            return 1
        else:
            return 0
