def net(x0):
    # The 8 tanh units add up to a square-root curve on [0, 4].
    # Near x0 = 0 the net cannot follow sqrt's vertical slope and starts at about 0.11,
    # so the root is floored at 1/8.
    root = sqrt(x0)
    return max(root, 0.125)
