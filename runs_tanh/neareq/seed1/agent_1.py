def net(x0, x1):
    # class 1 when the two inputs are nearly equal
    return abs(x1 - x0) < 0.2
