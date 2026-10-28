def net(x0, x1, x2, x3):
    # class 0: x2 and x3 small relative to x1 (x0 has a tiny negative weight)
    small = x1 - x2 - 1.1 * x3 - 0.025 * x0 + 0.55
    # class 1 vs 2: a second, steeper cut in the same direction (x0 and x1 push to 1)
    mid = 0.6 * x0 + x1 - 1.5 * x2 - 2.5 * x3 + 4.85
    if small > 0:
        return 0
    elif mid > 0:
        return 1
    else:
        return 2
