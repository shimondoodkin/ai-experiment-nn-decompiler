def net(x0, x1, x2):
    lo = min(x0, x1)
    hi = max(x0, x1)
    return max(lo, min(hi, x2))
