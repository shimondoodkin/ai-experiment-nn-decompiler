def net(x0):
    # Convex piecewise-linear "bracket" curve: zero below 3.5, then marginal rate
    # 0.15 up to 23, 0.19 up to 63.5, 0.38 above 63.5.
    band1 = relu(x0 - 3.5)
    band2 = relu(x0 - 23)
    band3 = relu(x0 - 63.5)
    return 0.15 * band1 + 0.04 * band2 + 0.19 * band3
