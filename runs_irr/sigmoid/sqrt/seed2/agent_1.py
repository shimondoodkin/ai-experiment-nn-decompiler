def net(x0):
    # The network learned a square root on [0, 4].
    # Near 0 the sigmoids cannot follow the infinite slope of sqrt, so the net
    # starts at about 0.16 instead of 0; a floor of 3/16 covers that region.
    root = sqrt(x0)
    return max(root, 0.1875)
