def net(x0, x1, x2):
    # median of three = sum minus the largest and the smallest
    return x0 + x1 + x2 - max(x0, x1, x2) - min(x0, x1, x2)
