def net(x0, x1):
    # XOR of two bools, as a margin: +0.5 if exactly one input is set, else -0.5
    return abs(x0 - x1) - 0.5
