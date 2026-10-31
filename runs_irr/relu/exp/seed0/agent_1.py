def net(x0):
    # convex piecewise-linear curve: three lines, the upper envelope wins
    low = 0.4 * x0 + 0.9
    mid = 2 * x0 + 0.7
    high = 5 * x0 - 3
    return max(low, mid, high)
