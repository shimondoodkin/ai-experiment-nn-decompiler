def net(x0):
    # Band / window detector: true when x0 lies strictly inside (-1/2, 1/2),
    # i.e. abs(x0) < 1/2.
    half_width = 0.5
    return -half_width < x0 < half_width
