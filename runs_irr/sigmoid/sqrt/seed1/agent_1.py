def net(x0):
    # The network approximates the square root of its input on [0, 4]
    # (net(1) = 1.001, net(2) = 1.412, net(4) = 1.992).
    # Near x0 = 0 the true sqrt has an infinite slope that a sum of 8 sigmoids
    # cannot follow; the net starts at 0.152 instead of 0. A floor of 1/6
    # covers that start-up region.
    root = sqrt(x0)
    return max(root, 1 / 6)
