def net(x0):
    # The network approximates the square root on [0, 4].
    # A ReLU net cannot follow sqrt's infinite slope at 0; it starts at about 0.05
    # instead, so the root is floored there.
    root = sqrt(x0)
    return max(root, 0.05)
