def net(x0):
    # Decreasing piecewise-linear curve with one kink at x0 = 1.25:
    # steep below the kink (slope -1.475), shallow above it (slope -0.175).
    knee = 1.25
    base = 0.88 - 0.175 * x0
    steep = 1.3 * relu(knee - x0)
    return base + steep
