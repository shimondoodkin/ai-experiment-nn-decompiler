def net(x0):
    # concave piecewise-linear curve on [0, 4]: slope 1.25, then 0.5 after x = 0.4, then 0.3 after x = 1.65
    start = 0.175
    bend1 = relu(x0 - 0.4)
    bend2 = relu(x0 - 1.65)
    return start + 1.25 * x0 - 0.75 * bend1 - 0.2 * bend2
