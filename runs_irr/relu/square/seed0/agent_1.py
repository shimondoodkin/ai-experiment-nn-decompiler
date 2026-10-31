def net(x0):
    # even function of x0: flat near 0, then slope 1.6 from |x0| = 0.4, slope 3.2 from |x0| = 1.2
    # (a piecewise-linear bowl; the knots 0.4 / 1.2 and slopes 1.6 / 3.2 match chords of x0*x0)
    a = abs(x0)
    floor_level = 0.05
    return floor_level + 1.6 * relu(a - 0.4) + 1.6 * relu(a - 1.2)
