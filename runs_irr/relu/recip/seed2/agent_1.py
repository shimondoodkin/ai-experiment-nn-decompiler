def net(x0):
    # Convex piecewise-linear under-approximation of roughly 1/x0 on [0.5, 4].
    # Three lines, each the previous one rescaled by x -> 2x, y -> y/2
    # (intercept halves, slope quarters); kinks near x0 = 10/11 and 20/11.
    steep = 3 - 2.2 * x0
    middle = 1.5 - 0.55 * x0
    flat = 0.75 - 0.1375 * x0
    return max(steep, middle, flat)
