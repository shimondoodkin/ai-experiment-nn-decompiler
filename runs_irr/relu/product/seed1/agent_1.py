def net(x0, x1):
    # The network approximates the product x0 * x1 with ReLU kinks along the
    # two diagonals (x0*x1 = ((x0+x1)^2 - (x0-x1)^2) / 4).
    # It undershoots the product in the far corners, where it saw few points:
    # the (2,2)/(-2,-2) corners (|x0+x1| > 3) and the (2,-2) corner (x0-x1 > 3).
    total = x0 + x1
    diff = x0 - x1
    product = x0 * x1
    corner_sag = 0.2 * relu(abs(total) - 3) - 0.1 * relu(diff - 3)
    return product - corner_sag
