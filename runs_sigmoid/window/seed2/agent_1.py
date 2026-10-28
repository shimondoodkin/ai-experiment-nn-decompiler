def net(x0):
    # class 1 when x0 lies inside the band (-0.5, 0.5), i.e. |x0| < 1/2
    inside = abs(x0) < 0.5
    return 1 if inside else -1
