def net(x0, x1, x2):
    # median of three inputs = total minus the largest minus the smallest
    lo = min(x0, x1, x2)
    hi = max(x0, x1, x2)
    median = x0 + x1 + x2 - lo - hi
    return median
