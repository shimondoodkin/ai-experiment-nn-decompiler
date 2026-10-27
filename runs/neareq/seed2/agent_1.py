def net(x0, x1):
    # class 1 iff the two inputs are close: |x0 - x1| < 0.2
    return 0.2 - abs(x0 - x1)
