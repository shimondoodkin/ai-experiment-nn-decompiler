def net(x0):
    # The net is a convex piecewise-linear fit of the parabola x0^2 (6 live ReLUs,
    # knots about 0.64 apart). Its pieces sit slightly below x0^2 at the knots and
    # slightly above between them; the shift -1/80 centres that ripple.
    square = x0 * x0
    return square - 0.0125
