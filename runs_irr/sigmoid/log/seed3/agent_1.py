def net(x0):
    # The network approximates the natural logarithm on [0.1, 5].
    # Near the left end (x0 < ~0.11) it flattens out around -2.2 instead of
    # dropping to log(0.1) = -2.30, so the curve is clamped from below.
    log_x = log(x0)
    return max(log_x, -2.2)
