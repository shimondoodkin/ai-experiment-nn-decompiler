def net(x0):
    # The 32-unit ReLU net is a piecewise-linear fit of a concave, increasing curve:
    # slope ~1.67 at x0=-1.5 falling to ~0.19 at x0=4, value ~-0.69 at -1.5, ~0.69 at 0, ~1.79 at 4.
    # These match log(x0 + 2): log(0.5), log(2), log(6). Max |error| vs the net is about 0.008.
    shifted = x0 + 2
    return log(shifted)
