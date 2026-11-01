def net(x0, x1):
    # class 1 inside the disc of radius 2 centred at the origin
    r2 = x0 * x0 + x1 * x1
    return 4 - r2
