def net(x0, x1):
    # |x0 - x1|-like valley with different slopes on each side of the diagonal
    above = relu(x1 - 0.875 * x0 - 0.25)        # x1 above the diagonal
    above_steep = relu(0.85 * x1 - x0 - 0.25)   # second kink: steeper further above
    below = relu(x0 - x1 + 0.125)               # x1 below the diagonal
    # plateau step in x0: rises from x0 = -0.7, levels off near x0 = -(x1 + 1) / 6
    x0_step = 0.9 * relu(x0 + 0.7) - 0.95 * relu(x0 + (x1 + 1) / 6)
    # plateau step in x1: rises as x1 drops below ~1/3, levels off below ~-0.4
    x1_step = 0.7 * relu(1 / 3 - x1 - 0.075 * x0) - 0.85 * relu(-x1 - 0.15 * x0 - 0.4)
    return 0.8 * above + 0.35 * above_steep + 0.75 * below + x0_step + x1_step
