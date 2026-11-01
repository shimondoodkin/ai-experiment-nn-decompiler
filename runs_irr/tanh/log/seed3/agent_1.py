def net(x0):
    # the net learned log(x0); near the low edge (x0 < ~0.11) it flattens,
    # so log is clamped from below there
    return max(log(x0), -2.2)
