def net(x0, x1):
    # A radial bump: peak about 1.06 at the origin, falling to about 0 beyond radius about 1.2.
    # Each unit is relu(direction . x + offset), scaled so that the largest coefficient is 1.
    # Cone: kinks through (nearly) the origin, all with negative weights -> the tip of the bump.
    c0 = relu(-x0 - 0.55 * x1 - 0.05)
    c1 = relu(0.95 * x0 + x1 - 0.15)
    c2 = relu(-0.15 * x0 + x1 + 0.05)
    c3 = relu(-0.45 * x0 + x1 + 0.05)
    c4 = relu(x0 - 0.8 * x1)
    c5 = relu(-x0 - 0.25 * x1 + 0.05)
    c6 = relu(-0.1 * x0 - x1 + 0.05)
    c7 = relu(-x0 + 0.4 * x1 + 0.05)
    cone = sum(-0.432 * c0, -0.131 * c1, -0.249 * c2, -0.065 * c3, -0.277 * c4, -0.144 * c5, -0.419 * c6, -0.354 * c7)
    # Inner ring: active inside a line at distance about 1.2 from the origin (positive weights).
    i0 = relu(-x0 + 1.2)
    i1 = relu(0.7 * x0 + x1 + 1.4)
    i2 = relu(x0 - 0.35 * x1 + 1.4)
    i3 = relu(-0.6 * x0 - x1 + 1.4)
    inner = sum(0.422 * i0, 0.249 * i1, 0.131 * i2, 0.358 * i3)
    # Outer ring: active outside a line at distance about 1.2 from the origin (positive weights).
    o0 = relu(-0.25 * x0 - x1 - 1.4)
    o1 = relu(0.65 * x0 - x1 - 1.4)
    o2 = relu(-0.55 * x0 + x1 - 1.4)
    o3 = relu(-x0 - 0.1 * x1 - 1.2)
    outer = sum(0.143 * o0, 0.333 * o1, 0.324 * o2, 0.319 * o3)
    return cone + inner + outer - 0.425
