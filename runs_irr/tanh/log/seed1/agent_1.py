def net(x0):
    # The network computes the natural logarithm of its input.
    # Near the lower bound (x0 < 0.11) the net flattens out compared with log, so the input
    # is floored at 0.11 there (log(0.11) = -2.207, the net gives -2.225 at x0 = 0.1).
    x = max(x0, 0.11)
    return log(x)
