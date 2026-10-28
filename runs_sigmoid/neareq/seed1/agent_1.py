def net(x0, x1):
    # class 1 when the two inputs are close: |x0 - x1| < 0.2
    diff = x1 - x0
    return 0.2 - abs(diff)
