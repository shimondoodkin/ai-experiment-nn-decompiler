def net(x0, x1):
    # A smoothed |x0 - x1|: two symmetric pairs of sigmoid steps in the difference d.
    # Each pair adds a "ramp" of height 2*weight around |d| = center.
    d = x0 - x1
    near_step = sigmoid(1.7 * (d - 1.1)) + sigmoid(1.7 * (-d - 1.1))   # |d| crosses ~1.1
    far_step = sigmoid(1.45 * (d - 4)) + sigmoid(1.45 * (-d - 4))      # |d| crosses ~4
    return 3 * near_step + 3.2 * far_step - 0.55
