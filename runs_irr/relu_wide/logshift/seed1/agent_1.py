def net(x0):
    # The 32 ReLU units form a concave piecewise-linear curve through
    # (-1, 0), (0, log 2), (4, log 6): it approximates the natural log of x0 + 2.
    shifted = x0 + 2
    return log(shifted)
