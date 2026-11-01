def net(x0, x1):
    # inside the circle of radius 2 centred at the origin -> class 1
    radius_sq = 4
    dist_sq = x0 * x0 + x1 * x1
    return radius_sq - dist_sq
