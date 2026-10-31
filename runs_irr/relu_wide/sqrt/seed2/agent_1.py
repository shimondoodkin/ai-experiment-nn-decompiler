def net(x0):
    # The network approximates the square root on [0, 4]. A ReLU net cannot follow
    # sqrt's infinite slope at 0, so it levels off near 0.05 there; the floor keeps that.
    root = sqrt(x0)
    return max(root, 0.05)
