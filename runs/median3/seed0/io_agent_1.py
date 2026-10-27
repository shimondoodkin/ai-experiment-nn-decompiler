def net(x0, x1, x2):
    # median of three inputs
    low = min(x0, x1)
    high = max(x0, x1)
    median = max(low, min(high, x2))
    return median
