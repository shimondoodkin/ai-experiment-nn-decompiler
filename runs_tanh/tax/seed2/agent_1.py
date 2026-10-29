def net(x0):
    # Convex piecewise-linear ramp: gentle slope at first, steeper after the knee (~x0 = 77).
    low_rate = 0.17 * x0
    high_rate = 0.39 * x0 - 17
    return max(low_rate, high_rate)
