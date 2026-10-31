def net(x0):
    # The network computes the square root of x0 on [0, 4].
    # Near x0 = 0 the sigmoids cannot follow sqrt's infinite slope, so the net
    # flattens out at about 0.14; a constant floor of 0.15 models that.
    root = sqrt(x0)
    return max(root, 0.15)
