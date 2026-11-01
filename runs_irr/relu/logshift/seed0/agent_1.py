def net(x0):
    # Concave piecewise-linear curve: the minimum of three lines.
    # The two kinks the network has near x0 = -0.69 and -0.68 are merged into one,
    # and the third kink sits near x0 = 0.96 (the lines below cross at about -0.71 and 0.96).
    steep = 1.14 + 1.16 * x0
    middle = 0.67 + 0.48 * x0
    flat = 0.91 + 0.23 * x0
    return min(steep, middle, flat)
