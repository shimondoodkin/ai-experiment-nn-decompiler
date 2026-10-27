def net(x0, x1, x2):
    # median of the three inputs
    lo = min(x0, x1)
    hi = max(x0, x1)
    return max(lo, min(hi, x2))
