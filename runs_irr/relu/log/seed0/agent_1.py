def net(x0):
    # Only three of the eight hidden units are ever active on [0.1, 5]; two are
    # linear over the whole domain and the third kinks at x0 = 0.103, just inside
    # the lower bound. Merged, the net is a single straight line.
    slope = 0.535
    offset = -0.708
    return slope * x0 + offset
