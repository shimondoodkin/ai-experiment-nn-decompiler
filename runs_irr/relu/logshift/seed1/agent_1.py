def net(x0):
    # The 1 -> 8 -> 1 ReLU net is a piecewise-linear interpolation of a natural log
    # shifted so that the curve crosses zero at x0 = -1 (net(-1) = 0.00018).
    shifted = x0 + 2
    return log(shifted)
