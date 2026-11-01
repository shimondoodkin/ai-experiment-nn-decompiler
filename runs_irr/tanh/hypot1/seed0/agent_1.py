def net(x0, x1):
    # The net approximates the distance from (x0, x1, 0) to (0, 0, -1): sqrt(1 + x0^2 + x1^2).
    dist = sqrt(1 + x0 * x0 + x1 * x1)
    # It undershoots in the four corners of the box (about -0.08 at |x0| = |x1| = 2).
    corner_sag = 0.08 * relu(abs(x0) + abs(x1) - 3.4)
    return dist - corner_sag
