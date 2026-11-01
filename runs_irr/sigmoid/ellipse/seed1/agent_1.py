def net(x0, x1):
    # class 1 inside the ellipse centred at (0.5, 0) with semi-axes 2 (x0) and 1 (x1):
    # (x0 - 0.5)^2 / 4 + x1^2 < 1, scaled by 4 to keep integer coefficients
    dx = x0 - 0.5
    inside_margin = 4 - dx * dx - 4 * x1 * x1
    return inside_margin
