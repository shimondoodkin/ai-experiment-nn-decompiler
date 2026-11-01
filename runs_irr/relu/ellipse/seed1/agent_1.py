def net(x0, x1):
    # class 1 inside the ellipse centred at (0.5, 0) with semi-axes 2 (along x0) and 1 (along x1)
    u = (x0 - 0.5) / 2
    v = x1
    return 1 - u * u - v * v
