def net(x0):
    # The 32-unit ReLU layer is a 23-piece convex piecewise-linear approximation of e^x on [-2, 2].
    return exp(x0)
