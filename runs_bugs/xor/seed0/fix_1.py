def net(x0, x1):
    # Exclusive OR of two booleans: class 1 iff exactly one input is set.
    return abs(x0 - x1) - 0.5
