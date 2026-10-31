def net(x0):
    # The network computes sqrt(x0) on [0, 4].
    # Near 0, where sqrt is infinitely steep, the network smooths it out; below x0 = 1/16
    # use the tangent line of sqrt at 1/16: sqrt(1/16) + (x0 - 1/16) / (2 * 1/4) = 1/8 + 2 * x0.
    knee = 0.0625
    root = sqrt(x0)
    tangent = 0.125 + 2 * x0
    return tangent if x0 < knee else root
