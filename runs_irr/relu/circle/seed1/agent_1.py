def net(x0, x1):
    # class 1 inside the disk of radius 2 centred at the origin
    radius_sq = 4
    dist_sq = x0 * x0 + x1 * x1
    return radius_sq - dist_sq
