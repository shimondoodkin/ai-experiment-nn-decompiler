def net(x0, x1):
    # inside an axis-aligned ellipse centred at (0.5, 0), semi-axes 2 (x0) and 1 (x1)
    u = (x0 - 0.5) / 2
    return 1 - u * u - x1 * x1
