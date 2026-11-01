def net(x0):
    # Concave piecewise-linear curve (rises fast, then flattens; looks like a
    # sqrt-shaped approximation on [0, 4]). Written as the min of three lines.
    steep = 0.17 + 1.25 * x0      # x0 < ~0.41
    middle = 0.48 + 0.5 * x0      # ~0.41 < x0 < 1.7
    flat = 0.82 + 0.3 * x0        # x0 > 1.7
    return min(steep, middle, flat)
