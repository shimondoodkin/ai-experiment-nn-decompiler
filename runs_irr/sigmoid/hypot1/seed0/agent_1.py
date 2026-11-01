def net(x0, x1):
    # Radially symmetric bowl: the net approximates hypot(1, x0, x1) = sqrt(1 + r^2).
    # The 0.98 factor accounts for the net flattening slightly toward the box corners
    # (sigmoid saturation); the exact formula sqrt(1 + r2) misses only at |x0|,|x1| ~ 2.
    r2 = x0 * x0 + x1 * x1
    return sqrt(1 + 0.98 * r2)
