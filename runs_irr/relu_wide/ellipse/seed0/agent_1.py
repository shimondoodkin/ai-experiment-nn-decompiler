def net(x0, x1):
    # Inside an axis-aligned ellipse centred at (0.5, 0), semi-axes 2 (along x0) and 1 (along x1):
    # ((x0 - 0.5) / 2)^2 + x1^2 < 1, scaled by 4 so the score is a plain polynomial.
    dx = x0 - 0.5
    return 4 - dx * dx - 4 * x1 * x1
