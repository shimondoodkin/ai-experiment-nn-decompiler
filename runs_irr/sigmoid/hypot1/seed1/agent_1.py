def net(x0, x1):
    # The network approximates the "lifted distance" sqrt(1 + x0^2 + x1^2)
    # (1 at the origin, about 2.24 at the edge midpoints, about 3 at the corners),
    # but its fitting error (up to 0.09 at the (-2, -2) corner) is larger than the
    # 2% check tolerance, so the program below keeps a pruned, snapped sigmoid basis.
    # Each unit is a soft step that bends the bowl up towards one side of the square.
    up_x1 = sigmoid(x1 - 1)                          # rise towards x1 = +2
    up_x1_left = sigmoid(-0.5 * x0 + x1 - 1.9)       # rise towards x1 = +2, more on the left
    up_x0 = sigmoid(2 * x0 + 0.25 * x1 - 2.7)        # rise towards x0 = +2
    up_low = sigmoid(-0.45 * x0 - 1.1 * x1 - 1.25)   # rise towards x1 = -2 (and the (-2, -2) corner)
    up_x0_low = sigmoid(1.5 * x0 - 1.25 * x1 - 2.1)  # rise towards the (+2, -2) corner
    up_left = sigmoid(-1.45 * x0 - 0.45 * x1 - 2.6)  # rise towards x0 = -2
    bowl = 2.5 * up_x1 + 2.45 * up_x1_left + 1.2 * up_x0 + 2.13 * up_low + 0.84 * up_x0_low + 1.82 * up_left
    return bowl - 0.72 + 0.18 * x0 - 0.21 * x1
