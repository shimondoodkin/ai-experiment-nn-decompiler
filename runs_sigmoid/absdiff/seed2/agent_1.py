def net(x0, x1):
    # Smooth |x1 - x0|: the net depends almost only on the difference d = x1 - x0.
    d = x1 - x0
    total = x0 + x1
    # d < 0 side: one wide sigmoid ramp (~ -d for -4 < d < -1, saturating near 5.2).
    left_ramp = 5.9 * sigmoid(-0.8 * d - 1.9)
    # d > 0 side: three staggered sigmoid steps centred near d = 0.86, 2.2, 4.1
    # that together climb ~ +d and saturate near 5.4.
    right_ramp = 1.7 * sigmoid(3.5 * d - 3) + 1.47 * sigmoid(2 * d - 4.4) + 2.98 * sigmoid(1.4 * d - 5.7)
    # Small correction (<= 0.2) along the diagonal: slightly higher where x0 + x1 is small.
    diag_fix = 0.13 * sigmoid(2 - 2 * total) + 0.08 * sigmoid(2 + 2 * total)
    return left_ramp + right_ramp + diag_fix - 0.695
