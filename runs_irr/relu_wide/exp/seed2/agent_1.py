def net(x0):
    # The 32-unit ReLU layer is a piecewise-linear fit of the exponential on [-2, 2].
    return exp(x0)
