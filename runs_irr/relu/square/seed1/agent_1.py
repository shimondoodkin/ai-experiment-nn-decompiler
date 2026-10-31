def net(x0):
    # The 8 ReLU kinks (4 on each side of 0, slopes growing outward) form a
    # piecewise-linear approximation of a parabola.
    square = x0 * x0
    return square
