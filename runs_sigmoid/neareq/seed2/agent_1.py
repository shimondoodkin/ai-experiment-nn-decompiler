def net(x0, x1):
    # class 1 when x0 and x1 are nearly equal: |x0 - x1| < 0.2
    gap = abs(x0 - x1)
    return 0.2 - gap
