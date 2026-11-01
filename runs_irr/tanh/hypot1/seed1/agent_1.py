def net(x0, x1):
    # a bowl with its minimum near the origin (about 1.06), rising to about 3 at the corners,
    # built from three tanh ridges in directions roughly 120 degrees apart
    left_up = tanh(0.3 * x0 + 0.5 * x1 - 0.87)
    left_down = tanh(0.3 * x0 - 0.5 * x1 - 0.87)
    right = tanh(-0.6 * x0 - 0.8)
    return 2.05 * (left_up + left_down) + 1.9 * right + 5.2
