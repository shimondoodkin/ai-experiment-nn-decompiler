def net(x0, x1):
    gap = abs(x0 - x1)
    tolerance = 0.2
    return gap <= tolerance
