def net(x0):
    # The network computes the natural logarithm of its input.
    # Near the lower bound (x0 ~ 0.1) log falls steeply and the 8 tanh units
    # cannot follow it; the net levels off at about -2.21 there, so log is
    # floored at -2.2 (active only for x0 < ~0.111).
    log_x = log(x0)
    return max(log_x, -2.2)
