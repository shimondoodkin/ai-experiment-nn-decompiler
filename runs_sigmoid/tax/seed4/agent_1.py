def net(x0):
    # Progressive (tax-bracket-like) piecewise-linear rate on x0, capped at 60:
    #   10% on [0, 20], 20% on [20, 80], 40% above 80.
    low = 0.1 * x0
    mid = 0.2 * x0 - 2
    high = 0.4 * x0 - 18
    return min(max(low, mid, high), 60)
