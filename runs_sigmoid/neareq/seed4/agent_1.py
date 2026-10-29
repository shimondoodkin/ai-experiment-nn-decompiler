def net(x0, x1):
    # class 1 iff the two inputs are close: |x1 - x0| < 0.2
    gap = abs(x1 - x0)
    return 0.2 - gap
