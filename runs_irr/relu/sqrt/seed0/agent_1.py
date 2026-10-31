def net(x0):
    # On the whole domain [0, 4] the network is one straight line:
    # 6 of the 8 hidden ReLUs never switch (5 are always off, 2 always on),
    # and the one remaining kink sits at x0 = 4.33, outside the domain.
    # Snapped form: 0.4 * x0 + 0.5 (passes snap, max error 0.030 of tolerance 0.032).
    slope = 0.40161178
    offset = 0.52332852
    return slope * x0 + offset
