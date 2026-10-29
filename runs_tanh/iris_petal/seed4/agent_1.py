def net(x0, x1):
    # Three ordered classes separated by two straight lines in the (x0, x1) plane.
    # class 0: small x0 and x1 (below the steep line x0 + 2.4*x1 = 4.2)
    # class 2: large x0 and x1 (above the diagonal x0 + x1 = 6.6)
    # class 1: the band in between
    size_low = x0 + 2.4 * x1
    size_high = x0 + x1
    if size_low < 4.2:
        return 0
    if size_high < 6.6:
        return 1
    return 2
