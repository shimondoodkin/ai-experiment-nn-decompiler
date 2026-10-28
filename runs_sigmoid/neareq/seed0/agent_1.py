def net(x0, x1):
    # class 1 when the two inputs are within 0.2 of each other
    gap = abs(x1 - x0)
    return gap < 0.2
