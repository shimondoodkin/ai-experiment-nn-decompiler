def net(x0):
    # The 32-unit ReLU layer is a piecewise-linear approximation of the reciprocal:
    # convex, decreasing, f(0.5)~2, f(1)~1, f(2)~0.5, f(4)~0.25.
    reciprocal = 1 / x0
    return reciprocal
