def net(x0):
    # The network approximates the natural logarithm of its input.
    # Near the low end of the domain (x0 < ~0.15) log is too steep for the
    # sigmoid units, so the net flattens out; clamping the argument at 0.12
    # reproduces that.
    x = max(x0, 0.12)
    return log(x)
